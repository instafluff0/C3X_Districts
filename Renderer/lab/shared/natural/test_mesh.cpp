// Literal pre-extraction production expressions: an independent byte/query oracle.
// These protect the current render contract, not a retired experiment.
#include "mesh.h"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace c3x_renderer::fidelity;
using Vertex=MapVertex;
using Layers=std::array<std::vector<Vertex>,25>;

template<class Height,class Shore,class Cancelled>
bool reference_relief(NaturalData const&natural,int real,Tile owner,GroundProjection project_natural,
    Height height_natural,Shore shore_sample_at,Cancelled cancelled,Layers&natural_vertices) {
    int nc=project_natural.column,nr=project_natural.row;
    struct {int real_terrain_type;}tile{real};
    auto triangle=[](std::vector<Vertex>&out,Vertex const&a,Vertex const&b,Vertex const&c){out.push_back(a);out.push_back(b);out.push_back(c);};
    if(tile.real_terrain_type==5){
        Hill hill=composed_hill(owner);std::uint32_t state=hill.seed;
        constexpr unsigned cells[]={0,0,0,0,1,1,1,2,2,2};
        for(unsigned ordinal=0;ordinal<10;ordinal++){
            float keep=random01(state),angle=random01(state)*6.283185307f;
            float radius=std::sqrt(random01(state))*.22f,phase=random01(state)*6.283185307f,scale=.90f+.20f*random01(state);
            if(keep>hill.rockiness)continue;
            float cu=.5f+std::cos(phase)*radius,cv=.5f+std::sin(phase)*radius;
            auto point=[&](unsigned x,unsigned y){
                float du=(x/8.f-.5f)*.42f*scale,dv=(y/8.f-.5f)*.37f*scale;
                float u=cu+std::cos(angle)*du-std::sin(angle)*dv,v=cv+std::sin(angle)*du+std::cos(angle)*dv;
                float support=0;float h=height_natural(float(nc)+u,float(nr)+1-v,&support)+.45f;
                auto out=project_natural(float(nc)+u,float(nr)+1-v,h);
                out.u=(float(cells[ordinal])+.0012f+x/8.f*(1-2*.0012f))*.25f;
                out.v=(.0012f+y/8.f*(1-2*.0012f))*.25f;
                out.material_grass=std::max(0.f,(h-2.5f)/112);out.material_plains=2;out.material_desert=angle;
                auto shore=shore_sample_at(float(nc)+u,float(nr)+1-v);
                out.base_terrain=-10+coast_coverage(float(shore.distance),float(shore.beach_width));
                return out;
            };
            for(unsigned y=0;y<8;y++)for(unsigned x=0;x<8;x++){
                auto a=point(x,y),b=point(x+1,y),c=point(x+1,y+1),d=point(x,y+1);
                triangle(natural_vertices[1],a,b,c);triangle(natural_vertices[1],a,c,d);
            }
        }
    }
    if(tile.real_terrain_type==6){
        unsigned variant=mountain_seed(owner)%5u;
        auto const&hf=natural.fields[natural.macro[variant][0]];
        auto const&bf=natural.fields[natural.macro[variant][1]];
        std::vector<Vertex> grid(128*128);
        for(unsigned y=0;y<128;y++){
            if(cancelled())return false;
            for(unsigned x=0;x<128;x++){
                float u=x/127.f,v=y/127.f,h=hf.sample(u,v),du=1/127.f;
                auto out=project_natural(float(nc)+.5f+(u-.5f)*1.85f,float(nr)+.5f+(.5f-v)*1.55f,2.5f+h*165);
                float n[]={-(hf.sample(u+du,v)-hf.sample(u-du,v))*(165/112.f)/(2*du*1.85f),
                    (hf.sample(u,v+du)-hf.sample(u,v-du))*(165/112.f)/(2*du*1.55f),1};normalize3(n);
                out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=u;out.v=v;
                out.material_grass=h;out.material_plains=2;out.material_desert=bf.sample(u,v);out.base_terrain=42;
                grid[y*128+x]=out;
            }
        }
        for(unsigned y=0;y<127;y++)for(unsigned x=0;x<127;x++){
            auto&a=grid[y*128+x];auto&b=grid[y*128+x+1];auto&c=grid[(y+1)*128+x+1];auto&d=grid[(y+1)*128+x];
            if(std::max({a.material_desert,b.material_desert,c.material_desert,d.material_desert})<.015f)continue;
            triangle(natural_vertices[2],a,b,c);triangle(natural_vertices[2],a,c,d);
        }
    }

    return true;
}
template<class Height,class Shore,class River,class Hash,class Random,class Cancelled>
bool reference_forest(NaturalData const&natural,Tile owner,GroundProjection project_natural,
    std::vector<BuildingBounds> const&buildings,Height height_natural,Shore shore_sample_at,
    River river_at,Hash hash,Random random,Cancelled cancelled,Layers&natural_vertices) {
    int nc=project_natural.column,nr=project_natural.row;
        std::uint32_t seed=std::uint32_t(owner.source_x*0x193u)^std::uint32_t(owner.source_y*0x217u);
        unsigned density=31+hash(seed^0xa53du)%11u;
        for(unsigned i=0;i<density;i++){
            if(cancelled())return false;
            unsigned selected=hash(seed+i*31u)%180;Recipe const*recipe=nullptr;
            for(auto const&r:natural.recipes){if(selected<r.count){recipe=&r;break;}selected-=r.count;}
            if(!recipe)return false;
            float ring=std::sqrt((float(i)+.5f)/float(density));
            float angle=2.39996323f*float(i)+random(seed^0x71b3u)*6.283185307f;
            float u=.5f+std::cos(angle)*ring*.43f,v=.5f+std::sin(angle)*ring*.43f;
            float scale=recipe->scale*(1+recipe->variation*(random(seed+i*71u+23u)*2-1))*.46f;
            float yaw=random(seed+i*97u+47u)*6.283185307f;
            float co=std::cos(yaw),si=std::sin(yaw);auto const&body=natural.bodies[recipe->object];
            auto const&mat=natural.materials[body.material];
            // Clip flags are authoritative source metadata. Test the body hull,
            // not just its center, against captured building and water geometry.
            float x0=1e9f,y0=1e9f,x1=-1e9f,y1=-1e9f;
            for(auto const&p:body.vertices){float x=float(nc)+u+(p.position[0]*co-p.position[1]*si)*scale;
                float y=float(nr)+1-v-(p.position[0]*si+p.position[1]*co)*scale;
                x0=std::min(x0,x);x1=std::max(x1,x);y0=std::min(y0,y);y1=std::max(y1,y);}
            bool clipped=false;
            for(auto const&b:buildings)if(x0<b.x1 && x1>b.x0 && y0<b.y1 && y1>b.y0)clipped=true;
            // Conservative footprint bounds against the authoritative fields:
            // a center-only or sparse point test can miss a channel between
            // samples. Signed distance minus a containing radius cannot.
            float cx=(x0+x1)*.5f,cy=(y0+y1)*.5f,rx=(x1-x0)*.5f,ry=(y1-y0)*.5f;
            float radius=std::hypot(rx,ry);
            if(shore_sample_at(cx,cy).distance<radius)clipped=true;
            float screen_radius=std::hypot((rx+ry)*64,(rx+ry)*32);
            if(river_at(cx,cy)<9+std::max(screen_radius,radius*64))clipped=true;
            if(clipped)continue;
            float ground_h=height_natural(float(nc)+u,float(nr)+1-v);
            // Uniform source XYZ scale precedes the documented source-to-world
            // basis conversion; normals use its inverse transpose.
            constexpr float z_basis=150.f/(.82f*64.f);
            for(auto const&p:body.vertices){
                float x=float(nc)+u+(p.position[0]*co-p.position[1]*si)*scale;
                float y=float(nr)+1-v-(p.position[0]*si+p.position[1]*co)*scale;
                auto out=project_natural(x,y,ground_h+p.position[2]*scale*z_basis*112);
                float n[]={p.normal[0]*co-p.normal[1]*si,-(p.normal[0]*si+p.normal[1]*co),p.normal[2]/z_basis};normalize3(n);
                out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=p.uv[0];out.v=p.uv[1];
                out.material_grass=1;out.material_plains=mat.repeat?2.f:0.f;out.material_desert=mat.channels[3]!=0xffffffffu?1.f:0.f;
                out.material_marsh=mat.channels[4]!=0xffffffffu?1.f:0.f;out.authored_relief_height=float(mat.tint);
                out.authored_relief_blend=mat.channels[6]!=0xffffffffu?1.f:0.f;
                out.base_terrain=mat.repeat?41.f:40.f;
                natural_vertices[3+recipe->object].push_back(out);
            }
        }
    return true;
}

struct Observation {unsigned kind;float x,y;};
struct Probe {
    std::vector<Observation> observations;
    unsigned cancellations=0,stop=0;
    int mode;
    float height(float x,float y,float*support=nullptr) {
        observations.push_back({0,x,y});if(support)*support=.25f;
        return 2.5f+x*.17f+y*.23f;
    }
    struct Shore {float distance,beach_width;};
    Shore shore(float x,float y) {
        observations.push_back({1,x,y});
        return {mode==2 ? .1f : 100.f,.2f};
    }
    double river(float x,float y) {
        observations.push_back({2,x,y});
        return mode==3 ? 8.999999999 : 1000.;
    }
    bool cancelled(){return ++cancellations==stop;}
};
unsigned hash(unsigned value) {
    value^=value>>16;value*=0x7feb352du;value^=value>>15;value*=0x846ca68bu;
    return value^(value>>16);
}
float random_value(unsigned value){return float(hash(value)&0x00ffffffu)/16777215.f;}
std::size_t check(Layers const&a,Layers const&b,Probe const&x,Probe const&y) {
    std::size_t count=0;
    for(unsigned i=0;i<a.size();i++){
        assert(a[i].size()==b[i].size());
        if(!a[i].empty())assert(!std::memcmp(a[i].data(),b[i].data(),a[i].size()*sizeof(Vertex)));
        count+=a[i].size();
    }
    assert(x.cancellations==y.cancellations && x.observations.size()==y.observations.size());
    for(unsigned i=0;i<x.observations.size();i++){
        auto left=x.observations[i],right=y.observations[i];
        assert(left.kind==right.kind && left.x==right.x && left.y==right.y);
    }
    return count;
}
int main(){
    NaturalData data;data.fields.resize(10);
    for(unsigned i=0;i<10;i++){
        auto&f=data.fields[i];f.width=17;f.height=13;f.minimum=.07f;f.maximum=.93f;
        for(unsigned y=0;y<f.height;y++)for(unsigned x=0;x<f.width;x++)
            f.pixels.push_back(std::uint8_t((x*17+y*13+i*43)%256));
    }
    for(unsigned i=0;i<5;i++){data.macro[i][0]=i*2;data.macro[i][1]=i*2+1;}
    data.materials.resize(22);data.bodies.resize(22);
    for(unsigned i=0;i<22;i++){
        auto&m=data.materials[i];m.repeat=i%2;m.tint=(i/2)%2;
        for(unsigned c=0;c<7;c++)m.channels[c]=(i+c)%2?0:0xffffffffu;
        auto&b=data.bodies[i];b.material=i;
        b.vertices={{{-.25f,-.2f,0},{.2f,.4f,.8f},{0,0}},
                    {{.25f,-.2f,.1f},{.8f,.1f,.4f},{1,0}},
                    {{0,.3f,.5f},{.1f,.8f,.2f},{.5f,1}}};
    }
    for(unsigned i=0;i<25;i++)data.recipes.push_back({i%22,.4f+i*.01f,.2f,i==24?12u:7u,0,0,i%8,.1f,.2f});
    std::size_t vertices=0,observations=0;unsigned scopes=0;bool variants[5]={};
    for(int seed=0;seed<20;seed++)for(int real:{2,5,6,7})for(int mode=0;mode<4;mode++){
        int c=seed-10,r=3-seed;Tile owner{seed*2,seed*4,c,r,real};
        variants[mountain_seed(owner)%5]=true;
        GroundProjection projection{c,r,seed%2?32.f:64.f,seed%2?16.f:32.f,
            (seed%2?64.f:128.f)/224*.82f,seed%3?480.f:640.f};
        std::vector<BuildingBounds> buildings;
        if(mode==1)buildings.push_back({float(c)-2,float(r)-2,float(c)+2,float(r)+2});
        Layers actual,expected;Probe a{{},0,0,mode},b{{},0,0,mode};
        auto height_a=[&](float x,float y,float*s=nullptr){return a.height(x,y,s);};
        auto height_b=[&](float x,float y,float*s=nullptr){return b.height(x,y,s);};
        auto shore_a=[&](float x,float y){return a.shore(x,y);};
        auto shore_b=[&](float x,float y){return b.shore(x,y);};
        auto cancel_a=[&](){return a.cancelled();};auto cancel_b=[&](){return b.cancelled();};
        assert(emit_relief_meshes(data,real,owner,projection,height_a,shore_a,cancel_a,actual[1],actual[2]));
        assert(reference_relief(data,real,owner,projection,height_b,shore_b,cancel_b,expected));
        if(real==7){
            assert(emit_forest(data,owner,projection,buildings,height_a,shore_a,
                [&](float x,float y){return a.river(x,y);},hash,random_value,cancel_a,actual));
            assert(reference_forest(data,owner,projection,buildings,height_b,shore_b,
                [&](float x,float y){return b.river(x,y);},hash,random_value,cancel_b,expected));
            if(mode==1 || mode==3)for(unsigned i=3;i<25;i++)assert(actual[i].empty());
        }
        vertices+=check(actual,expected,a,b);observations+=a.observations.size();scopes++;
    }
    for(bool seen:variants)assert(seen);
    for(int real:{6,7})for(unsigned stop:{1u,3u,20u}){
        Tile owner{2,4,3,-1,real};GroundProjection projection{3,-1,64,32,.4f,480};
        Layers actual,expected;Probe a{{},0,stop,0},b{{},0,stop,0};
        auto ha=[&](float x,float y,float*s=nullptr){return a.height(x,y,s);};
        auto hb=[&](float x,float y,float*s=nullptr){return b.height(x,y,s);};
        auto sa=[&](float x,float y){return a.shore(x,y);};
        auto sb=[&](float x,float y){return b.shore(x,y);};
        auto ca=[&](){return a.cancelled();};auto cb=[&](){return b.cancelled();};
        bool got,old;
        if(real==6){
            got=emit_relief_meshes(data,real,owner,projection,ha,sa,ca,actual[1],actual[2]);
            old=reference_relief(data,real,owner,projection,hb,sb,cb,expected);
        }else{
            got=emit_forest(data,owner,projection,{},ha,sa,[&](float x,float y){return a.river(x,y);},hash,random_value,ca,actual);
            old=reference_forest(data,owner,projection,{},hb,sb,[&](float x,float y){return b.river(x,y);},hash,random_value,cb,expected);
        }
        assert(!got && got==old);check(actual,expected,a,b);scopes++;
    }
    std::cout<<"PASS shared natural meshes: "<<scopes<<" scopes, "<<vertices<<" byte-identical vertices, "
             <<observations<<" ordered observations; five mountain variants, clipping and cancellation\n";
}
