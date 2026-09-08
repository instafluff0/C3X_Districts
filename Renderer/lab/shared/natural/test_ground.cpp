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
            struct Shore{float distance,beach_width;};
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
            expected.base_terrain=-10+coast_coverage(.36f,.2f)*remaining;
            expected.material_marsh=.4f/std::max(remaining,.00001f);
            expected.authored_relief_height=.2f/std::max(remaining,.00001f);
            expected.authored_relief_blend=.3f/std::max(remaining,.00001f);
            assert(std::memcmp(&got,&expected,sizeof(got))==0);
            comparisons++;
        }
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
    std::cout<<"PASS shared ground: "<<comparisons<<" exact vertices, 289 corners, 512 triangles, cancellation and coast clipping\n";
}
