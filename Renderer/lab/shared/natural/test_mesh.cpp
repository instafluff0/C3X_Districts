// Literal pre-extraction production expressions: an independent byte/query oracle.
// These protect the current render contract, not a retired experiment.
#include "mesh.h"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace c3x_renderer::fidelity;
using Vertex=MapVertex;
using Layers=std::array<std::vector<Vertex>,25>;

template<class Lookup,class Height,class Shore,class River,class Weights,class Cancelled>
bool reference_relief(NaturalData const&natural,int real,Tile owner,GroundProjection project_natural,
    Lookup lookup_natural,Height height_natural,Shore shore_sample_at,
    River river_at,Weights material_weights_for,Cancelled cancelled,Layers&natural_vertices) {
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
    {
        struct MountainPiece {unsigned height_field,blend_field;float center_x,center_y,long_span,cross_span,height_scale;bool connected,range_y;};
        std::vector<MountainPiece> pieces;
        for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
            int pc=nc+dc,pr=nr+dr;Tile piece_owner=lookup_natural(pc,pr);
            if(piece_owner.real!=6)continue;
            bool west=lookup_natural(pc-1,pr).real==6,east=lookup_natural(pc+1,pr).real==6;
            bool north=lookup_natural(pc,pr-1).real==6,south=lookup_natural(pc,pr+1).real==6;
            unsigned along_x=unsigned(west)+unsigned(east),along_y=unsigned(north)+unsigned(south);
            bool connected=along_x+along_y>0,turn=connected&&along_x==along_y;
            unsigned variant=mountain_seed(piece_owner)%5u;
            pieces.push_back({natural.macro[variant][0],natural.macro[variant][1],
                float(pc)+.5f+.09f*(int(east)-int(west)),
                float(pr)+.5f+.09f*(int(south)-int(north)),
                connected?(turn?2.08f:2.46f):1.85f,
                connected?(turn?1.82f:1.34f):1.55f,
                connected?142.f:165.f,connected,along_y>along_x});
        }
        struct MountainSample {float displacement=0,dominant=0,height=0,blend=0,u=0,v=0;};
        auto mountain_at=[&](float world_x,float world_y){
            MountainSample result;
            for(auto const&piece:pieces){
                float source_x=piece.range_y?(world_y-piece.center_y)/piece.long_span:
                    (world_x-piece.center_x)/piece.long_span;
                float source_y=piece.range_y?(world_x-piece.center_x)/piece.cross_span:
                    (world_y-piece.center_y)/piece.cross_span;
                float u=.5f+source_x,v=.5f-source_y;
                if(u<0||u>1||v<0||v>1)continue;
                float h=natural.fields[piece.height_field].sample(u,v);
                float blend=natural.fields[piece.blend_field].sample(u,v);
                float shaped=piece.connected?std::pow(std::max(0.f,h),.80f):h;
                float displacement=shaped*piece.height_scale*smooth01((blend-.28f)/.34f);
                if(displacement>result.dominant){
                    result.dominant=displacement;result.height=h;result.u=u;result.v=v;
                }
                if(displacement>0){
                    if(result.displacement<=0)result.displacement=displacement;
                    else{
                        float high=std::max(result.displacement,displacement);
                        float ridge=std::max(0.f,10.f-std::abs(result.displacement-displacement));
                        result.displacement=high+ridge*ridge/40.f;
                    }
                }
                result.blend=std::max(result.blend,blend);
            }
            return result;
        };
        if(!pieces.empty()){
            constexpr unsigned count=65,span=67;constexpr float step=1/64.f;
            constexpr unsigned river_count=19;
            std::array<float,river_count*river_count> river_scales;
            for(unsigned y=0;y<river_count;y++)for(unsigned x=0;x<river_count;x++){
                float px=float(nc)+(float(x)-1)/16.f;
                float py=float(nr)+1-(float(y)-1)/16.f;
                river_scales[y*river_count+x]=smooth01((float(river_at(px,py))-6.f)/16.f);
            }
            auto mountain_river_scale=[&](float world_x,float world_y){
                float gx=std::clamp((world_x-float(nc))*16+1,0.f,17.9999f);
                float gy=std::clamp((float(nr)+1-world_y)*16+1,0.f,17.9999f);
                unsigned x=unsigned(std::floor(gx)),y=unsigned(std::floor(gy));
                float tx=gx-x,ty=gy-y;
                float a=river_scales[y*river_count+x]*(1-tx)+river_scales[y*river_count+x+1]*tx;
                float b=river_scales[(y+1)*river_count+x]*(1-tx)+river_scales[(y+1)*river_count+x+1]*tx;
                return a*(1-ty)+b*ty;
            };
            std::vector<float> surface_height(span*span);
            for(unsigned y=0;y<span;y++){
                if(cancelled())return false;
                for(unsigned x=0;x<span;x++){
                    float world_x=float(nc)+(int(x)-1)*step;
                    float world_y=float(nr)+1-(int(y)-1)*step;
                    auto sample=mountain_at(world_x,world_y);
                    auto shore=shore_sample_at(world_x,world_y);
                    float scale=coast_relief(float(shore.distance),float(shore.beach_width))*
                        mountain_river_scale(world_x,world_y);
                    surface_height[y*span+x]=height_natural(world_x,world_y,nullptr)+
                        sample.displacement*scale;
                }
            }
            std::vector<Vertex> grid(count*count);
            for(unsigned y=0;y<count;y++){
                if(cancelled())return false;
                for(unsigned x=0;x<count;x++){
                    float world_x=float(nc)+x*step,world_y=float(nr)+1-y*step;
                    auto sample=mountain_at(world_x,world_y);
                    auto shore=shore_sample_at(world_x,world_y);
                    float relief=coast_relief(float(shore.distance),float(shore.beach_width));
                    float river_scale=mountain_river_scale(world_x,world_y);
                    sample.blend*=relief*river_scale;
                    unsigned at=(y+1)*span+x+1;
                    float elevation=surface_height[at];
                    auto out=project_natural(world_x,world_y,elevation);
                    float mountain_height=sample.displacement*relief*river_scale;
                    out.world_valid=1+std::max(0.f,(elevation-mountain_height-2.5f)/112);
                    float n[]={-(surface_height[at+1]-surface_height[at-1])/(2*step*112),
                        (surface_height[at+span]-surface_height[at-span])/(2*step*112),1};normalize3(n);
                    out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=sample.u;out.v=sample.v;
                    out.material_grass=sample.height;out.material_plains=2;out.material_desert=sample.blend;
                    auto weights=material_weights_for(world_x,world_y);
                    float source_weight=std::clamp(1-weights[3],0.f,1.f);
                    float desert_weight=std::clamp(weights[2]/std::max(source_weight,.00001f),0.f,1.f);
                    float coverage=coast_coverage(float(shore.distance),float(shore.beach_width))*(1-desert_weight)+
                        desert_coast_coverage(float(shore.distance))*desert_weight;
                    for(auto&weight:weights)weight/=std::max(source_weight,.00001f);
                    out.material_marsh=weights[4];out.authored_relief_height=weights[1];
                    out.authored_relief_blend=weights[2];
                    out.base_terrain=42+coverage*source_weight;
                    grid[y*count+x]=out;
                }
            }
            for(unsigned y=0;y+1<count;y++)for(unsigned x=0;x+1<count;x++){
                auto&a=grid[y*count+x];auto&b=grid[y*count+x+1];auto&c=grid[(y+1)*count+x+1];auto&d=grid[(y+1)*count+x];
                triangle(natural_vertices[2],a,b,c);triangle(natural_vertices[2],a,c,d);
            }
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
std::array<float,5> weights(float,float){return {.20f,.25f,.15f,.10f,.30f};}
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
    std::array<float,20> open_mountain_peaks{},river_valley_peaks{};
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
        auto topology=[&](int tc,int tr){
            int feature=real;
            if(real==6 && (tc!=c || tr!=r)){
                bool neighbor=mode==1?(tr==r && std::abs(tc-c)==1):
                    mode==2?(tc==c && std::abs(tr-r)==1):
                    mode==3?((tc==c+1&&tr==r)||(tc==c&&tr==r+1)):false;
                feature=neighbor?6:2;
            }
            return Tile{tc+tr,tc-tr,tc,tr,feature};
        };
        auto cancel_a=[&](){return a.cancelled();};auto cancel_b=[&](){return b.cancelled();};
        assert(emit_relief_meshes(data,real,owner,projection,topology,height_a,shore_a,
            [&](float x,float y){return a.river(x,y);},weights,cancel_a,actual[1],actual[2]));
        assert(reference_relief(data,real,owner,projection,topology,height_b,shore_b,
            [&](float x,float y){return b.river(x,y);},weights,cancel_b,expected));
        if(real==6)assert(!actual[2].empty());
        if(real==6 && !actual[2].empty()){
            float x0=1e9f,x1=-1e9f,y0=1e9f,y1=-1e9f;
            for(auto const&v:actual[2]){x0=std::min(x0,v.world_x);x1=std::max(x1,v.world_x);
                y0=std::min(y0,v.world_y);y1=std::max(y1,v.world_y);}
            assert(x0>=float(c) && x1<=float(c+1));
            assert(y0>=float(r) && y1<=float(r+1));
            auto const&v=actual[2].front();
            assert(v.base_terrain>=42.f && v.base_terrain<=42.90001f);
            assert(std::abs(v.authored_relief_height-.25f/.9f)<1e-5f);
            assert(std::abs(v.authored_relief_blend-.15f/.9f)<1e-5f);
            assert(std::abs(v.material_marsh-.30f/.9f)<1e-5f);
            float peak=0;
            for(auto const&vertex:actual[2])peak=std::max(peak,vertex.world_z);
            if(mode==0)open_mountain_peaks[unsigned(seed)]=peak;
            if(mode==3)river_valley_peaks[unsigned(seed)]=peak;
        }
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
    for(unsigned i=0;i<open_mountain_peaks.size();i++){
        assert(open_mountain_peaks[i]>0);
        assert(river_valley_peaks[i]>0);
        assert(river_valley_peaks[i]<open_mountain_peaks[i]*.35f);
    }
    {
        auto topology=[](int c,int r){return Tile{c+r,c-r,c,r,c==1&&r==0?6:2};};
        auto emit_patch=[&](int c,int real){
            Layers out;Probe probe{{},0,0,0};GroundProjection projection{c,0,64,32,.4f,480};
            auto height=[&](float x,float y,float*s=nullptr){return probe.height(x,y,s);};
            auto shore=[&](float x,float y){return probe.shore(x,y);};
            auto river=[&](float x,float y){return probe.river(x,y);};
            auto cancel=[&](){return probe.cancelled();};
            assert(emit_relief_meshes(data,real,topology(c,0),projection,topology,
                height,shore,river,weights,cancel,out[1],out[2]));
            return out[2];
        };
        auto left=emit_patch(0,2),right=emit_patch(1,6);
        assert(!left.empty()&&!right.empty());
        struct Edge {float z=-1,nx=0,ny=0,nz=0;};
        std::array<Edge,65> a{},b{};
        auto collect=[](std::vector<Vertex>const&vertices,float x,std::array<Edge,65>&edge){
            for(auto const&v:vertices)if(std::abs(v.world_x-x)<1e-6f){
                int i=int(std::lround((1-v.world_y)*64));
                if(i>=0&&i<65)edge[unsigned(i)]={v.world_z,v.normal_x,v.normal_y,v.normal_z};
            }
        };
        collect(left,1,a);collect(right,1,b);unsigned matches=0;
        for(unsigned i=0;i<65;i++)if(a[i].z>=0&&b[i].z>=0){
            assert(a[i].z==b[i].z&&a[i].nx==b[i].nx&&a[i].ny==b[i].ny&&a[i].nz==b[i].nz);matches++;
        }
        assert(matches>=8);scopes+=2;
    }
    for(int real:{6,7})for(unsigned stop:{1u,3u,20u}){
        Tile owner{2,4,3,-1,real};GroundProjection projection{3,-1,64,32,.4f,480};
        Layers actual,expected;Probe a{{},0,stop,0},b{{},0,stop,0};
        auto ha=[&](float x,float y,float*s=nullptr){return a.height(x,y,s);};
        auto hb=[&](float x,float y,float*s=nullptr){return b.height(x,y,s);};
        auto sa=[&](float x,float y){return a.shore(x,y);};
        auto sb=[&](float x,float y){return b.shore(x,y);};
        auto topology=[&](int tc,int tr){return Tile{tc+tr,tc-tr,tc,tr,tc==3&&tr==-1?real:2};};
        auto ca=[&](){return a.cancelled();};auto cb=[&](){return b.cancelled();};
        bool got,old;
        if(real==6){
            got=emit_relief_meshes(data,real,owner,projection,topology,ha,sa,
                [&](float x,float y){return a.river(x,y);},weights,ca,actual[1],actual[2]);
            old=reference_relief(data,real,owner,projection,topology,hb,sb,
                [&](float x,float y){return b.river(x,y);},weights,cb,expected);
        }else{
            got=emit_forest(data,owner,projection,{},ha,sa,[&](float x,float y){return a.river(x,y);},hash,random_value,ca,actual);
            old=reference_forest(data,owner,projection,{},hb,sb,[&](float x,float y){return b.river(x,y);},hash,random_value,cb,expected);
        }
        assert(!got && got==old);check(actual,expected,a,b);scopes++;
    }
    std::cout<<"PASS shared natural meshes: "<<scopes<<" scopes, "<<vertices<<" byte-identical vertices, "
             <<observations<<" ordered observations; five mountain variants, unified surfaces and cancellation\n";
}
