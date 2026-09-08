#pragma once
#include <algorithm>
namespace c3x_renderer { namespace fidelity {
inline float coast_ramp(float value) {
    value=std::clamp(value,0.f,1.f);
    return value*value*(3-2*value);
}
inline float coast_coverage(float distance,float beach_width) {
    return coast_ramp((distance-beach_width-.06f)/.16f);
}
// The retained beach is flat. Finish the material fade before raising the
// replacement surface, so alpha clipping cannot expose an elevated dune edge.
// Both endpoint slopes are zero; authored inland relief is unchanged.
inline float coast_relief(float distance,float beach_width) {
    return coast_ramp((distance-beach_width-.22f)/.36f);
}
} }
