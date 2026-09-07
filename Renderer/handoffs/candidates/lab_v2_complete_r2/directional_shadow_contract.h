#pragma once
// Implementation-preparation contract. Not wired into production by this package.
// Positions and normals enter the canonical pickup world before lighting/shadows.
#include <array>
#include <cmath>
#include <stdexcept>

namespace c3x_shadow_contract {
using V3 = std::array<float, 3>;
inline V3 normalized(V3 v) {
    float n = std::hypot(v[0], std::hypot(v[1], v[2]));
    if (!std::isfinite(n) || n < 1e-8f) throw std::runtime_error("invalid shadow direction");
    for (auto& x : v) x /= n;
    return v;
}
// Same weighted direction, cancellation fallback and fixed slope as Lab Q6 and
// pickup-r1. Do not select a different dominant source per object category.
template<class Environment> V3 light(Environment const& e) {
    float x = e.sun_direction[0]*e.sun_intensity + e.moon_direction[0]*e.moon_intensity;
    float y = e.sun_direction[1]*e.sun_intensity + e.moon_direction[1]*e.moon_intensity;
    float h = std::hypot(x, y);
    if (!std::isfinite(h)) throw std::runtime_error("nonfinite environment");
    return h > 1e-6f ? normalized({x/h, y/h, 1.35f}) : normalized({-1, 0, 1.35f});
}
inline V3 project_to_plane(V3 p, V3 toward_light, float receiver_z) {
    if (toward_light[2] <= 0) throw std::runtime_error("light below receiver");
    float height = p[2] - receiver_z;
    return {p[0]-toward_light[0]/toward_light[2]*height,
            p[1]-toward_light[1]/toward_light[2]*height, receiver_z};
}
// Unit/resource local coordinates: sx=(x-y)*W/2; sy=(x+y)*H/2
// -z*150*W/224. Pickup: sx=(u+v)*W/2; sy=(u-v)*H/2
// -w*112*.82*W/224. Translation is supplied by authoritative anchors.
constexpr float local_height_to_world = 150.f/(112.f*.82f);
inline V3 posed_local_to_world(V3 p, V3 anchor = {0,0,0}) {
    return {anchor[0]+p[0], anchor[1]-p[1], anchor[2]+p[2]*local_height_to_world};
}
inline V3 posed_normal_to_world(V3 n) {
    // Inverse transpose of diag(1,-1,local_height_to_world), after pose/yaw.
    return normalized({n[0], -n[1], n[2]/local_height_to_world});
}
inline std::array<float,2> screen(V3 p, float tile_width, float tile_height) {
    return {(p[0]+p[1])*tile_width*.5f,
            (p[0]-p[1])*tile_height*.5f-p[2]*112.f*.82f*tile_width/224.f};
}
} // namespace c3x_shadow_contract
