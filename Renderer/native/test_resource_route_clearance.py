"""A layout that makes room for routes moves away from them and fits the space.

Roads and railroads are drawn through a tile's centre, so a centred oasis had a
road running over its pond. A composition carrying a "clearance/routes" marker
moves toward the side farthest from every drawn route point and shrinks to fit
the clear space (route_clearance_layout), staying inside its tile.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class ResourceRouteClearanceTests(unittest.TestCase):
    def test_group_moves_to_the_open_side_fits_and_stays_in_its_tile(self):
        run_cpp(r'''
#include "Renderer/native/render_core/resource_instances.h"
#include <cassert>
#include <cmath>
#include <vector>
using Points = std::vector<std::array<float,2>>;
Points line(float u0, float v0, float u1, float v1) {
    Points points;
    for (int i = 0; i <= 40; ++i) {float t = i/40.f; points.push_back({u0+(u1-u0)*t, v0+(v1-v0)*t});}
    return points;
}
float nearest(Points const& points, float u, float v) {
    float best = 1e9f;
    for (auto const& p : points) best = std::min(best, std::hypot(p[0]-u, p[1]-v));
    return best;
}
int main() {
    using c3x_renderer::render_core::route_clearance_layout;
    // A straight road along u through the centre: the group steps off it across v,
    // and its shrunken reach stays clear of the road.
    auto road = line(0, .5f, 1, .5f);
    auto across = route_clearance_layout(road, .45f, .35f, .6f, .32f);
    assert(std::abs(across[2]) > .2f);                                     // {shrink, du, dv}
    assert(nearest(road, .5f+across[1], .5f+across[2]) >= .45f*across[0] + .04f - 1e-4f);   // fits clear
    // Arms east (+u) and south (+v) meet at the centre: the group takes the open corner.
    auto corner = line(.5f, .5f, 1, .5f), south = line(.5f, .5f, .5f, 1);
    corner.insert(corner.end(), south.begin(), south.end());
    auto open = route_clearance_layout(corner, .45f, .35f, .6f, .32f);
    assert(open[1] < -.1f && open[2] < -.1f);
    // Around a tight loop (a railroad's passing loop) nothing fits: the smallest size.
    Points loop;
    for (int i = 0; i < 64; ++i) {float a = 6.2831853f*i/64; loop.push_back({.5f+.15f*std::cos(a), .5f+.15f*std::sin(a)});}
    auto tight = route_clearance_layout(loop, .45f, .35f, .6f, .32f);
    assert(std::abs(tight[0] - .35f) < 1e-5f);
    // Without routes nearby it keeps its largest size; every result stays in the tile.
    auto far = route_clearance_layout(line(0, 0, 1, 0), .45f, .35f, .6f, .32f);
    assert(std::abs(far[0] - .6f) < 1e-5f);
    for (auto layout : {across, open, tight, far})
        assert(std::abs(layout[1]) <= .47f - .45f*layout[0] + 1e-5f && std::abs(layout[2]) <= .47f - .45f*layout[0] + 1e-5f);
}
''')


if __name__ == "__main__":
    unittest.main()
