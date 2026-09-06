// Q4 geometry provider; the shared platform remains the only GPU backend.
#include "../../contracts/packet_v1.h"
#include "../../shared/environment_runtime.cpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
int main(int argc, char **argv) {
    if (argc != 7) return 2;
    std::ifstream f(argv[6]); std::stringstream s; s << f.rdbuf();
    std::smatch m; std::string desc=s.str();
    if (!std::regex_search(desc,m,std::regex("\"relief_packet\"\\s*:\\s*\"([^\"]+)\""))) return 3;
    auto p=labv2::read_packet(m[1].str().c_str());
    if (p.width!=unsigned(atoi(argv[2])) || p.height!=unsigned(atoi(argv[3]))) return 4;
    p.downsample=unsigned(atoi(argv[5]));
    auto e=c3x_renderer::evaluate_environment(float(atof(argv[4])),0);
    float c[]={e.sun_direction[0],e.sun_direction[1],e.sun_direction[2],e.sun_intensity,
        e.sun_color[0],e.sun_color[1],e.sun_color[2],e.exposure,
        e.moon_direction[0],e.moon_direction[1],e.moon_direction[2],e.moon_intensity,
        e.moon_color[0],e.moon_color[1],e.moon_color[2],e.night_activation,
        e.ambient_color[0],e.ambient_color[1],e.ambient_color[2],e.shadow_strength};
    memcpy(p.buffers[0].data(),c,sizeof(c));
    return labv2::write_packet(argv[1],p)?0:1;
}
