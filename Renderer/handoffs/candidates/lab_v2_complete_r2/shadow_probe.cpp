// Compare actual current native unit math with the canonical Lab/pickup frame.
#include "directional_shadow_contract.h"
#include "Renderer/native/environment_runtime.h"
#include "Renderer/native/unit_shadow.h"
#include "Renderer/terrain_lab/v2/systems/lighting/shadow_field_v1.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace c3x_shadow_contract;
    std::cout << "{\"classification\":\"coordinate probe; not rendered visual acceptance\",\"cases\":[";
    bool first=true; int compared=0;
    for (float hour : {0.f,6.f,12.f,18.f}) {
        auto e=c3x_renderer::evaluate_environment(hour,0);
        auto L=light(e);
        auto lab=q6::build_shadow_frame(std::vector<q6::WorldTriangle>{},e,64,6);
        for (int k=0;k<3;++k) assert(std::abs(L[k]-lab.L[k])<1e-6f);
        auto dominant=e.sun_intensity>=e.moon_intensity?e.sun_direction:e.moon_direction;
        c3x_renderer::UnitShadow old;
        assert(old.fit({{0,0,.5f}},dominant[0],dominant[1]));
        for (float zoom : {1.f,.5f}) for (int yaw=0;yaw<8;++yaw) {
            float a=yaw*3.14159265358979323846f/4;
            V3 p={.2f*std::cos(a),.2f*std::sin(a),.5f};
            auto world=posed_local_to_world(p);
            auto visual=screen(world,128*zoom,64*zoom);
            assert(std::abs(visual[0]-(p[0]-p[1])*64*zoom)<1e-5f);
            assert(std::abs(visual[1]-((p[0]+p[1])*32-p[2]*150*128/224)*zoom)<1e-5f);
            auto ground=screen(project_to_plane(world,L,0),128*zoom,64*zoom);
            auto base=screen({world[0],world[1],0},128*zoom,64*zoom);
            float dx=ground[0]-base[0],dy=ground[1]-base[1];
            float old_dx=(-old.dx+old.dy)*p[2]*64*zoom;
            float old_dy=(-old.dx-old.dy)*p[2]*32*zoom;
            // Projection must not acquire a new direction when an object turns.
            auto reference=screen(project_to_plane({0,0,world[2]},L,0),128*zoom,64*zoom);
            assert(std::hypot(dx-reference[0],dy-reference[1])<1e-4f);
            V3 tangent={.4f,.3f,.2f}, normal={-.3f,.4f,0};
            auto tw=posed_local_to_world(tangent),nw=posed_normal_to_world(normal);
            assert(std::abs(tw[0]*nw[0]+tw[1]*nw[1]+tw[2]*nw[2])<1e-6f);
            if(yaw==0) {
                if(!first)std::cout<<',';first=false;
                std::cout<<"{\"hour\":"<<hour<<",\"zoom\":"<<zoom
                    <<",\"canonical_ground_delta_px\":["<<dx<<','<<dy
                    <<"],\"current_unit_ground_delta_px\":["<<old_dx<<','<<old_dy<<"]}";
            }
            ++compared;
        }
    }
    c3x_renderer::EnvironmentState cancelled={};
    auto fallback=light(cancelled);assert(fallback[0]<0 && fallback[1]==0 && fallback[2]>0);
    std::cout<<"],\"pose_zoom_phase_checks\":"<<compared
        <<",\"lab_basis_matches\":true,\"production_shadow_unification_complete\":false}\n";
}
