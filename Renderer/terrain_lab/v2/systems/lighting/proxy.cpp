// Diagnostic packet only. Reuse the authoritative evaluator, never a category clock.
#include "../../contracts/packet_v1.h"
#include "../../shared/environment_runtime.cpp"
#include <cstdlib>
#include <cstring>
int main(int argc, char **argv) {
    if (argc != 7) return 2;
    labv2::Packet p;
    p.width = unsigned(atoi(argv[2])); p.height = unsigned(atoi(argv[3]));
    p.downsample = unsigned(atoi(argv[5]));
    auto e = c3x_renderer::evaluate_environment(float(atof(argv[4])), 0);
    float vertices[] = {-1,-1,.5f,0,0, 3,-1,.5f,2,0, -1,3,.5f,0,2};
    // float4-aligned, shared across every surface and caster in the witness.
    float constants[] = {
        e.sun_direction[0],e.sun_direction[1],e.sun_direction[2],e.sun_intensity,
        e.sun_color[0],e.sun_color[1],e.sun_color[2],e.exposure,
        e.moon_direction[0],e.moon_direction[1],e.moon_direction[2],e.moon_intensity,
        e.moon_color[0],e.moon_color[1],e.moon_color[2],e.night_activation,
        e.ambient_color[0],e.ambient_color[1],e.ambient_color[2],e.emissive_scale,
        float(p.width),float(p.height),0,0
    };
    p.buffers.resize(2);
    p.buffers[0].resize(sizeof(vertices)); memcpy(p.buffers[0].data(),vertices,sizeof(vertices));
    p.buffers[1].resize(sizeof(constants)); memcpy(p.buffers[1].data(),constants,sizeof(constants));
    labv2::Draw d; d.count=3; d.stride=20; d.constant_buffer=1;
    d.attributes={{3,0},{2,12}}; p.draws.push_back(d);
    return labv2::write_packet(argv[1],p) ? 0 : 1;
}
