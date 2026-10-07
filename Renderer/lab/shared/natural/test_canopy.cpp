// Canopy clearing and forest varieties through the shared production forest
// body. Visibility is checked in screen terms, independently of the clearing's
// own world-space test: a crown standing in front of a kept-clear point may not
// rise over it.
#include "mesh.h"
#include "patterns.h"
#include <cassert>
#include <cstring>
#include <iostream>
namespace c3x_renderer {
std::uint32_t stable_hash(std::uint32_t value){return patterns::feature_hash(value);}
float stable_random(std::uint32_t value){return patterns::stable_random(value);}
}
using namespace c3x_renderer::fidelity;
using Layers=std::array<std::vector<MapVertex>,3+max_natural_bodies>;
struct ShoreSample {float distance,beach_width;};
struct Tree {float x0,y0,x1,y1,height;unsigned body;};

// Each emitted tree is one body's vertices in order; recover hull and height.
std::vector<Tree> trees(NaturalData const&natural,Layers const&layers){
    constexpr float z_basis=150.f/(.82f*64.f);
    std::vector<Tree> out;
    for(unsigned body=0;body<natural.bodies.size();body++){
        auto const&v=layers[3+body];auto n=natural.bodies[body].vertices.size();
        assert(v.size()%n==0);
        for(std::size_t first=0;first<v.size();first+=n){
            Tree t{1e9f,1e9f,-1e9f,-1e9f,0,body};float z0=1e9f,z1=-1e9f;
            for(std::size_t i=first;i<first+n;i++){
                t.x0=std::min(t.x0,v[i].world_x);t.x1=std::max(t.x1,v[i].world_x);
                t.y0=std::min(t.y0,v[i].world_y);t.y1=std::max(t.y1,v[i].world_y);
                z0=std::min(z0,v[i].world_z);z1=std::max(z1,v[i].world_z);}
            t.height=(z1-z0)*112/(z_basis*112);out.push_back(t);
        }
    }
    return out;
}
std::vector<Tree> forest(NaturalData const&natural,int c,int r,CanopyClearing const&clearing){
    Tile owner{c+r,c-r,c,r,7};
    GroundProjection projection{c,r,64,32,128.f/224*.82f,640};
    Layers layers;
    bool ok=emit_forest(natural,owner,projection,{},[](float,float){return 2.5f;},
        [](float,float){return ShoreSample{100,.2f};},[](float,float){return 1000.f;},
        c3x_renderer::patterns::feature_hash,c3x_renderer::patterns::stable_random,[]{return false;},layers,clearing);
    assert(ok);
    return trees(natural,layers);
}
// The production vegetation-floor body; each patch's centre vertex (u=v=.5)
// identifies it. River distance is in 1/64 tile, to a channel along y=r+.3.
std::vector<std::array<float,2>> floor_patches(NaturalData const&natural,int c,int r,int real,CanopyClearing const&clearing){
    int nc=c,nr=r;Tile owner{c+r,c-r,c,r,real};
    struct {int real_terrain_type;} tile{real};
    GroundProjection project_natural{c,r,64,32,128.f/224*.82f,640};
    auto lookup_natural=[&](int tc,int tr){return Tile{tc+tr,tc-tr,tc,tr,real};};
    auto height_natural=[](float,float){return 2.5f;};
    auto shore_sample_at=[](float,float){return ShoreSample{100,.2f};};
    auto river_at=[&](float,float y){return std::abs(y-(float(r)+.3f))*64;};
    auto cancelled=[]{return false;};
    auto triangle=[](auto&out,auto const&a,auto const&b,auto const&c){out.push_back(a);out.push_back(b);out.push_back(c);};
    Layers natural_vertices;
    auto emit=[&]()->bool{
        #include "vegetation_floor_mesh_body.h"
        return true;
    };
    assert(emit());
    std::vector<std::array<float,2>> centres;
    for(auto const&v:natural_vertices[1])if(v.u==.5f && v.v==.5f &&
        std::none_of(centres.begin(),centres.end(),[&](auto const&p){return p[0]==v.world_x && p[1]==v.world_y;}))
        centres.push_back({v.world_x,v.world_y});
    return centres;
}
// A crown hides a ground point behind it when the point lies within the
// crown's screen width and below the hiding part of its screen height.
bool hides(Tree const&t,float x,float y,float fraction=CanopyClearing::hide_fraction){
    float cx=(t.x0+t.x1)*.5f,cy=(t.y0+t.y1)*.5f;
    float across=std::abs((cx+cy)-(x+y)),width=((t.x1-t.x0)+(t.y1-t.y0))*.5f;
    float up=(cx-cy)-(x-y);   // screen rows above the trunk, in x-y units
    return across<width*.8f && up>0 && up<fraction*t.height*150.f/32.f*.8f;
}
int main(){
    NaturalData natural;
    natural.materials.resize(23);natural.bodies.resize(23);
    for(unsigned i=0;i<23;i++){
        natural.bodies[i].material=i;
        // A small cone: a ground ring and a crown tip.
        auto&v=natural.bodies[i].vertices;
        for(unsigned k=0;k<6;k++){float a=float(k)*1.0471976f;
            v.push_back({{std::cos(a)*.3f,std::sin(a)*.3f,0},{0,0,1},{0,0}});
            v.push_back({{std::cos(a+.5f)*.3f,std::sin(a+.5f)*.3f,0},{0,0,1},{1,0}});
            v.push_back({{0,0,.6f},{0,0,1},{.5f,1}});}
    }
    for(unsigned i=0;i<25;i++)natural.recipes.push_back({i%22,.6f,.1f,i==24?12u:7u,0,0,3,0,0});
    for(unsigned i=25;i<35;i++)natural.recipes.push_back({i%22,.6f,.1f,i==34?13u:12u,0,0,3,0,0});

    // Optional varieties: one extra texture, one material, and a snow body
    // reusing body 0. Pine uses bodies 3..5; snow pine the new body 23.
    std::vector<std::uint8_t> file(8);std::memcpy(file.data(),"C3XFVAR1",8);
    auto u32=[&](unsigned value){file.insert(file.end(),reinterpret_cast<std::uint8_t*>(&value),reinterpret_cast<std::uint8_t*>(&value)+4);};
    auto bytes=[&](void const*data,std::size_t n){auto p=static_cast<std::uint8_t const*>(data);file.insert(file.end(),p,p+n);};
    u32(1);u32(1);u32(1);u32(2);
    std::string path="ab.dds";u32(unsigned(path.size()));bytes(path.data(),path.size());
    Material snow{};for(auto&channel:snow.channels)channel=0xffffffffu;snow.channels[0]=1;bytes(&snow,sizeof(snow));
    u32(0);u32(23);
    u32(3);for(unsigned body:{3u,4u,5u}){Recipe r{body,.6f,.1f,10,0,3,3,0,0};bytes(&r,sizeof(r));}
    u32(1);{Recipe r{23,.6f,.1f,9,0,3,3,0,0};bytes(&r,sizeof(r));}
    std::vector<int> textures(1);
    auto read=[](std::string const&name,std::vector<std::uint8_t>&out){
        if(name.find("ab.dds")==std::string::npos)return false;out.assign(200,0);return true;};
    auto upload=[](std::vector<std::uint8_t> const&,int&texture){texture=7;return true;};
    {
        // A truncated file changes nothing.
        NaturalData copy=natural;auto cut=file;cut.pop_back();
        assert(!copy.load_forest_varieties(cut,textures,read,upload,""));
        assert(copy.bodies.size()==23 && copy.recipes.size()==35 && copy.forest_sets[pine_forest].weight==0);
        assert(&copy.forest_set(pine_forest)==&copy.forest_sets[broadleaf_forest]);
    }
    assert(natural.load_forest_varieties(file,textures,read,upload,""));
    assert(natural.bodies.size()==24 && natural.bodies[23].material==23 && textures.size()==2 && textures[1]==7);
    assert(natural.bodies[23].vertices.size()==natural.bodies[0].vertices.size());
    assert(natural.forest_sets[pine_forest].first==35 && natural.forest_sets[pine_forest].weight==30);
    assert(natural.forest_sets[snow_pine_forest].first==38 && natural.forest_sets[snow_pine_forest].weight==9);

    unsigned trees_seen=0,kept=0,hidden_without=0,opened=0,cornered=0,closer=0,inside=0,animal=0;
    for(int c=-6;c<6;c++)for(int r=-6;r<6;r++){
        // A road across the tile in world x, sampled as the native boxes are.
        CanopyClearing road;std::vector<std::array<float,2>> centre;
        for(float x=float(c)-.5f;x<=float(c)+1.5f;x+=.04f){
            float y=float(r)+.5f+.2f*std::sin(x*3);
            road.boxes.push_back({x-.065f,y-.065f,x+.065f,y+.065f});centre.push_back({x,y});
        }
        auto open=forest(natural,c,r,{});
        auto cleared=forest(natural,c,r,road);
        assert(!cleared.empty());
        trees_seen+=unsigned(open.size());kept+=unsigned(cleared.size());
        for(auto const&t:cleared){
            for(auto const&b:road.boxes)assert(!(t.x0<b.x1 && t.x1>b.x0 && t.y0<b.y1 && t.y1>b.y0));
            for(auto const&p:centre)assert(!hides(t,p[0],p[1]));
        }
        for(auto const&t:open)for(auto const&p:centre)if(hides(t,p[0],p[1])){++hidden_without;break;}

        // A resource tile: a thinner stand, every trunk outside the opening.
        CanopyClearing resource;resource.open_x=float(c)+.5f;resource.open_y=float(r)+.5f;resource.open_radius=.25f;
        resource.boxes.push_back({float(c)+.3f,float(r)+.3f,float(c)+.7f,float(r)+.7f});
        auto stand=forest(natural,c,r,resource);
        assert(stand.size()<=12 && stand.size()<open.size());
        for(auto const&t:stand){
            float cx=(t.x0+t.x1)*.5f-resource.open_x,cy=(t.y0+t.y1)*.5f-resource.open_y;
            assert(std::hypot(cx,cy)>resource.open_radius);
            // The full spiral stays within .43 of the centre; the stand spreads to the corners.
            cornered+=std::hypot(cx,cy)>.48f;
            assert(t.x0>=float(c)-.2f && t.x1<=float(c)+1.2f && t.y0>=float(r)-.2f && t.y1<=float(r)+1.2f);
            assert(!(t.x0<float(c)+.7f && t.x1>float(c)+.3f && t.y0<float(r)+.7f && t.y1>float(r)+.3f));
        }
        opened+=unsigned(stand.size());

        // A stationary resource (plants, rocks) lets the stand close in to
        // .7 of the opening and crowns hide it with half the reach.
        CanopyClearing plant=resource;plant.boxes.clear();plant.open_scale=.7f;
        for(float d:{-.17f,.17f})plant.near_boxes.push_back({float(c)+.5f+d-.05f,float(r)+.5f+d-.05f,float(c)+.5f+d+.05f,float(r)+.5f+d+.05f});
        // Animated subjects keep the full clearance for the same parts.
        CanopyClearing herd=plant;herd.boxes.swap(herd.near_boxes);herd.open_scale=1;
        animal+=unsigned(forest(natural,c,r,herd).size());
        for(auto const&t:forest(natural,c,r,plant)){
            float cx=(t.x0+t.x1)*.5f-plant.open_x,cy=(t.y0+t.y1)*.5f-plant.open_y,d=std::hypot(cx,cy);
            assert(d>plant.open_radius*plant.open_scale);
            inside+=d-(t.x1-t.x0)*.5f<plant.open_radius;++closer; // crown edge within the full opening
            for(auto const&b:plant.near_boxes){
                assert(!(t.x0<b.x1 && t.x1>b.x0 && t.y0<b.y1 && t.y1>b.y0));
                assert(!hides(t,(b.x0+b.x1)*.5f,(b.y0+b.y1)*.5f,CanopyClearing::hide_fraction*.5f));
            }
        }

        // Varieties draw only their own recipe bodies.
        CanopyClearing pine;pine.variety=pine_forest;
        for(auto const&t:forest(natural,c,r,pine))assert(t.body>=3 && t.body<=5);
        CanopyClearing snowy;snowy.variety=snow_pine_forest;
        for(auto const&t:forest(natural,c,r,snowy))assert(t.body==23);
    }
    // Floor patches stay off the road, the resource and the river channel.
    unsigned patches=0,patches_on_road=0;
    for(int real:{7,8})for(int c=-4;c<4;c++)for(int r=-4;r<4;r++){
        CanopyClearing road;
        for(float x=float(c)-.5f;x<=float(c)+1.5f;x+=.04f)road.boxes.push_back({x-.045f,float(r)+.7f-.045f,x+.045f,float(r)+.7f+.045f});
        road.boxes.push_back({float(c)+.35f,float(r)+.35f,float(c)+.6f,float(r)+.6f});
        for(auto const&p:floor_patches(natural,c,r,real,road)){
            ++patches;
            for(auto const&b:road.boxes)assert(!(p[0]>b.x0 && p[0]<b.x1 && p[1]>b.y0 && p[1]<b.y1));
            assert(std::abs(p[1]-(float(r)+.3f))*64>9);
        }
        for(auto const&p:floor_patches(natural,c,r,real,{}))
            for(auto const&b:road.boxes)if(p[0]>b.x0 && p[0]<b.x1 && p[1]>b.y0 && p[1]<b.y1){++patches_on_road;break;}
    }
    assert(patches>50 && patches_on_road>20);
    // Without the clearing, crowns cover the road on most tiles.
    assert(hidden_without>100 && opened>0 && cornered>opened/4 && inside>0 && closer>animal);
    std::cout<<"canopy clearing: "<<trees_seen<<" trees, "<<kept<<" beside roads, "<<
        hidden_without<<" uncleared crowns hiding a road, "<<opened<<" around resources ("<<cornered<<" in corners); "<<
        patches<<" floor patches kept clear, "<<patches_on_road<<" uncleared on a road; "<<
        inside<<" of "<<closer<<" trees closer around stationary resources ("<<animal<<" around animals)\n";
}
