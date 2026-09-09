#pragma once
#include "../scene_lighting.h"
// Retained adapter for natural fixtures and the world source-shadow ABI.
namespace c3x_renderer { namespace fidelity {
inline std::array<float,12> light_frame(EnvironmentState const&e){
    return lighting::shadow_frame(e);
}
} }
