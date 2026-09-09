#include "ground.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
using namespace c3x_renderer::fidelity;

int main() {
    static_assert(offsetof(MapVertex,world_x)==120, "world layout");
    static_assert(offsetof(MapVertex,relief_owner_u)==152, "relief layout");
    unsigned comparisons=0;
    for(float zoom:{64.f,128.f})for(float target:{480.f,640.f})for(int column:{-17,0,29}) {
        GroundProjection project{column,-5,zoom*.5f,zoom*.25f,zoom/224.f*.82f,target};
        for(float u:{0.f,.375f,1.f})for(float v:{0.f,.8125f,1.f})for(float marsh:{0.f,.25f,1.f}) {
            auto height=[](float x,float y,float*support){if(support)*support=.125f;return 2.5f+x*3+y*2;};
            struct Shore{float distance,beach_width,rocky=0;};
            auto shore=[](float,float){return Shore{.36f,.2f};};
            auto weights=[&](float,float){return std::array<float,5>{{.1f,.2f,.3f,marsh,.4f}};};
            auto got=ground_surface(project,u,v,height,shore,weights);
            // Direct production formula: preserve every material/unused channel.
            MapVertex expected={};
            float x=float(column)+u,y=-5.f+1-v,h=height(x,y,nullptr);
            float dx=x-float(column),dy=y-(-5.f),base=(dx-dy+1)*project.half_height;
            expected.x=(dx+dy)*project.half_width;
            expected.y=base-(h-2.5f)*project.relief_scale;
            expected.z=base+(h-2.5f)*.0016f*target;
            expected.u=x;expected.v=y;expected.world_x=x;expected.world_y=y;
            expected.world_z=h/112;
            expected.world_valid=1+coast_ramp((.36f-.2f-.10f)/.90f);
            constexpr float e=.006f;
            float nx=-(height(x+e,y,nullptr)-height(x-e,y,nullptr))/(2*e*128);
            float ny=-(height(x,y+e,nullptr)-height(x,y-e,nullptr))/(2*e*128);
            float length=std::sqrt(nx*nx+ny*ny+1);
            expected.normal_x=nx/length;expected.normal_y=ny/length;expected.normal_z=1/length;
            expected.material_grass=std::max(0.f,(h-2.5f)/112);
            expected.material_plains=1;expected.material_desert=.125f;
            float remaining=std::clamp(1-marsh,0.f,1.f);
            float desert_weight=std::clamp(.3f/std::max(remaining,.00001f),0.f,1.f);
            float coverage=coast_coverage(.36f,.2f)*(1-desert_weight)+
                desert_coast_coverage(.36f)*desert_weight;
            expected.base_terrain=-10+coverage*remaining;
            expected.material_marsh=.4f/std::max(remaining,.00001f);
            expected.authored_relief_height=.2f/std::max(remaining,.00001f);
            expected.authored_relief_blend=.3f/std::max(remaining,.00001f);
            assert(std::memcmp(&got,&expected,sizeof(got))==0);
            comparisons++;
        }
    }
    // Supply a coastal rock weight with opaque coverage before elevation. The
    // shader selects steep faces; ordinary beach and inland markers stay zero.
    {
        GroundProjection project{0,0,64,32,1,480};
        float slope=400,rocky=1,distance=.2f;
        auto height=[&](float x,float,float*support){if(support)*support=0;return 2.5f+x*slope;};
        struct Shore{float distance,beach_width,rocky;};
        auto shore=[&](float,float){return Shore{distance,0,rocky};};
        auto weights=[](float,float){return std::array<float,5>{{1,0,0,0,0}};};
        auto sample=[&](){return ground_surface(project,.5f,.5f,height,shore,weights);};
        assert(sample().material_plains==1.25f);
        assert(sample().base_terrain>-9.2f);
        rocky=0;auto beach=sample();assert(beach.base_terrain>-9.2f);
        assert(beach.material_plains==1);
        rocky=1;slope=0;assert(sample().base_terrain==-9);
        slope=400;distance=.6f;assert(sample().base_terrain==-9);
        assert(sample().material_plains==1);
        distance=.04f;assert(sample().base_terrain==-9);
        distance=0;assert(sample().base_terrain==-10);
    }
    unsigned samples=0,rows=0;
    std::vector<MapVertex> mesh;
    auto surface=[&](float u,float v){samples++;MapVertex p={};p.u=u;p.v=v;p.base_terrain=-9;return p;};
    assert(emit_ground_grid(mesh,surface,[&](){rows++;return false;}));
    assert(samples==289 && rows==17 && mesh.size()==1536);
    for(unsigned y=0;y<16;y++)for(unsigned x=0;x<16;x++) {
        auto p=mesh.data()+(y*16+x)*6;
        assert(p[0].u==x/16.f && p[0].v==y/16.f);
        assert(p[1].u==(x+1)/16.f && p[1].v==y/16.f);
        assert(p[2].u==(x+1)/16.f && p[2].v==(y+1)/16.f);
        assert(std::memcmp(p,p+3,sizeof(MapVertex))==0);
        assert(std::memcmp(p+2,p+4,sizeof(MapVertex))==0);
        assert(p[5].u==x/16.f && p[5].v==(y+1)/16.f);
    }
    auto original=mesh;
    rows=0;
    assert(!emit_ground_grid(mesh,surface,[&](){return ++rows==3;}));
    assert(mesh.size()==original.size() && std::memcmp(mesh.data(),original.data(),mesh.size()*sizeof(MapVertex))==0);
    mesh.clear();
    assert(emit_ground_grid(mesh,[](float,float){MapVertex p={};p.base_terrain=-10;return p;},[]{return false;}));
    assert(mesh.empty());
    // The narrow cliff rise needs more than the ordinary two-sample span.
    // Compare triangle interpolation with the continuous curve at its midpoint.
    auto coast=[](float u,float){MapVertex p={};p.u=u;p.base_terrain=-9;
        float t=std::clamp((u-.04f)/.14f,0.f,1.f);p.y=46.6667f*t*t*(3-2*t);return p;};
    auto error=[&](unsigned divisions){
        std::vector<MapVertex> refined;assert(emit_ground_grid(refined,coast,[]{return false;},divisions));
        float worst=0;
        for(unsigned i=0;i<divisions;i++){
            auto const*tri=refined.data()+i*6;
            float expected=coast((i+.5f)/divisions,0).y;
            worst=std::max(worst,std::abs((tri[0].y+tri[1].y)*.5f-expected));
        }
        return worst;
    };
    assert(error(48)<error(16)*.25f);
    // Indexed production emission must reproduce every triangle byte, including
    // coast holes, append offsets, empty coverage and a cancelled second mesh.
    for(unsigned divisions:{16u,48u,64u})for(unsigned mode:{0u,1u,2u}){
        auto sample=[&](float u,float v){auto p=coast(u,v);p.v=v;
            p.world_x=u;p.world_y=v;p.base_terrain=mode==2 || (mode==1 && u<.5f && v<.5f)?-10.f:-9.f;return p;};
        std::vector<MapVertex> expanded,indexed;
        std::vector<unsigned> indices;
        for(unsigned repeat=0;repeat<2;repeat++){
            assert(emit_ground_grid(expanded,sample,[]{return false;},divisions));
            assert(emit_ground_grid(indexed,sample,[]{return false;},divisions,&indices));
            assert(indices.size()==expanded.size());
            std::vector<bool> used(indexed.size());
            for(unsigned i=0;i<indices.size();i++){
                assert(indices[i]<indexed.size());used[indices[i]]=true;
                assert(std::memcmp(&indexed[indices[i]],&expanded[i],sizeof(MapVertex))==0);
            }
            for(bool value:used)assert(value);
        }
        if(mode==0)assert(indexed.size()==2*(divisions+1)*(divisions+1));
        if(mode==2)assert(indexed.empty() && indices.empty());
        auto before=indexed;auto before_indices=indices;unsigned calls=0;
        assert(!emit_ground_grid(indexed,sample,[&](){return ++calls==3;},divisions,&indices));
        assert(indices==before_indices && indexed.size()==before.size());
        if(!before.empty())assert(std::memcmp(indexed.data(),before.data(),before.size()*sizeof(MapVertex))==0);
    }
    std::cout<<"PASS shared ground: "<<comparisons<<" exact vertices, 289 corners, 512 triangles, cancellation and coast clipping\n";
}
