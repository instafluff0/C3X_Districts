"""Baked resource ground decals do not climb steep faces.

Composition ground decals follow the terrain vertex by vertex. On a mountain the
upper half of a decal near the foot climbed the steep face and stretched into a
tall smear (the "stretched gold" on mountains). decal_ground keeps a decal on
ground that rises no more than half its distance from the decal's centre; the
face hides the rest. Gentle relief and falling ground are followed exactly, and a
decal centred up a steep face (which would stretch down to its foot) is left out.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class ResourceDecalSlopeTests(unittest.TestCase):
    def test_decal_follows_gentle_ground_but_not_a_steep_face(self):
        source = (ROOT / "Renderer/native/object_compiler.h").read_text()
        start = source.index("inline float decal_ground(")
        helper = source[start:source.index("\n}\n", source.index("inline bool steep_decal(", start)) + 3]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cmath>
''' + helper + r'''
int main() {
    float per_tile = 150.f / .82f;                 // relief units per tile
    float centre = 10.f;
    // Gentle rise (a hill, 1:4) and falling ground are followed exactly.
    assert(decal_ground(centre + .1f * per_tile * .25f, centre, .1f) == centre + .1f * per_tile * .25f);
    assert(decal_ground(centre - .1f * per_tile, centre, .1f) == centre - .1f * per_tile);
    // A steep face (1:1) is clamped to a 1:2 rise from the centre.
    float clamped = decal_ground(centre + .1f * per_tile, centre, .1f);
    assert(std::abs(clamped - (centre + .05f * per_tile)) < 1e-4f);
    assert(decal_ground(centre + 1.f, centre, 0.f) == centre);   // the centre itself never rises
    // A decal centred up a steep face (ground falling 1:1 below it) is left out;
    // one on a hillside (1:4) is kept.
    assert(steep_decal(.1f * per_tile, .1f) && !steep_decal(.1f * per_tile * .25f, .1f));
}
''')


if __name__ == "__main__":
    unittest.main()
