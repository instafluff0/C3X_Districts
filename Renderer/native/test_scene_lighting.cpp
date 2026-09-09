#include "scene_lighting.h"
#include "source_fidelity/light_frame.h"
#include "unit_shadow.h"
#include <cassert>
#include <iostream>
using namespace c3x_renderer;
using Vec=std::array<float,3>;
float dot(Vec a,Vec b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
int main() {
    float invalid_light[3]={0,0,0};
    assert(lighting::ground_offset(invalid_light,1)==(std::array<float,2>{0,0}));
    unsigned phases=0;
    for(int season=0;season<4;++season)for(int minute=0;minute<=1440;++minute) {
        float hour=minute/60.f;
        auto e=evaluate_environment(hour,season),next=evaluate_environment(hour+.001f,season);
        auto key=lighting::key_light(e),next_key=lighting::key_light(next);
        assert(dot(key.direction,next_key.direction)>.9999f);
        for(int i=0;i<3;++i)assert(std::abs(key.color[i]-next_key.color[i])<.001f);
        auto basis=fidelity::light_frame(e);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            Vec a={basis[4*i],basis[4*i+1],basis[4*i+2]};
            Vec b={basis[4*j],basis[4*j+1],basis[4*j+2]};
            assert(std::abs(dot(a,b)-(i==j?1.f:0.f))<1e-5f);
        }
        std::vector<UnitShadow::Point> points={{0,0,.4f},{.2f,0,.4f},{0,.2f,.4f}};
        UnitShadow unit;assert(unit.fit(points,key.direction[0],key.direction[1]));
        unit.triangle(points[0],points[1],points[2]);
        auto p=unit.project(points[0]);
        auto offset=lighting::ground_offset(key.direction.data(),.4f*lighting::object_height_to_world);
        assert(std::abs(offset[0]-p[0])<1e-5f && std::abs(offset[1]+p[1])<1e-5f);
        assert(std::abs(std::hypot(offset[0],offset[1])-.4f*lighting::object_height_to_world/lighting::shadow_slope)<1e-5f);
        // Verify screen direction independently of world/local sign conventions.
        float phase=hour*3.14159265358979323846f/12;
        for(float zoom:{.5f,1.f,1.5f,2.f}) {
            float sx=(p[0]-p[1])*64*zoom,sy=(p[0]+p[1])*32*zoom;
            float length=std::hypot(sx,sy);
            assert(std::abs(sx/length-std::cos(phase))<1e-5f);
            assert(std::abs(sy/length+std::sin(phase))<1e-5f);
        }
        ++phases;
    }
    // A source sloping plane's transformed normal must remain perpendicular
    // to BOTH transformed tangents, for every actor orientation and height scale.
    for(int angle=0;angle<360;angle+=5)for(float height_scale:{.5f,1.f,lighting::object_height_to_world,3.f}) {
        float c=std::cos(angle*.01745329252f),s=std::sin(angle*.01745329252f);
        auto n=lighting::object_normal(-c-s,-s+c,1,height_scale);
        Vec t0={c,-s,height_scale},t1={-s,-c,-height_scale};
        assert(std::abs(dot(n,t0))<1e-5f && std::abs(dot(n,t1))<1e-5f);
    }
    std::cout<<"PASS shared lighting: "<<phases<<" phases; world/unit screen direction, height, zoom, normals and continuous handover\n";
}
