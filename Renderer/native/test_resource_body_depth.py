"""Animated resource bodies take the terrain's depth basis.

Animated bodies are drawn without a content projection, so the natural depth
correction that fixed static resource bodies never reached them. Their depth
grew by 0.75 * relief per unit of ground height while natural terrain grows by
0.0016 * view height, so on raised ground a band about as tall as the ground
height lay behind the terrain: legs, lowered heads and small animals vanished
(the original "cattle with their legs cut off"). VSResourceBody now uses the
natural basis. This test evaluates the shader's depth statement: a body point
at the ground must match the terrain's depth there at any ground height, and
points above it must sort in front.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class ResourceBodyDepthTests(unittest.TestCase):
    def test_body_at_ground_matches_terrain_depth_at_any_height(self):
        shader = (ROOT / "Renderer/lab/shared/shaders/objects/resource_skinning.hlsl").read_text()
        start = shader.index("    precise float depth=resource_anchor.y")
        depth = shader[start:shader.index(";", start) + 1].replace("precise ", "")
        run_cpp(r'''
#include <cassert>
#include <cmath>
#include <initializer_list>
struct float4 { float x, y, z, w; };
float depth_of(float ground, float lift_tiles, float local_x, float local_y) {
    float4 resource_anchor{0, 400.f - ground * 1.14f * .82f, 0, 0};   // CPU centre already raised by the ground
    float4 resource_offset{0, 0, 0, ground};
    float4 resource_projection{64.f, 32.f, 1.14f, 1500.f};
    struct { float x, y, z; } local{local_x, local_y, lift_tiles};
    float relief = resource_projection.z * .82f;
    float feature_height = local.z * 150.f / .82f;
''' + depth + r'''
    return depth;
}
int main() {
    for (float ground : {0.f, 5.f, 20.f, 60.f}) {
        float base = 400.f + (.1f + .05f) * 32.f;                         // foot point's ground screen y
        float terrain = base + ground * .0016f * 1500.f;                  // natural terrain depth there
        assert(std::abs(depth_of(ground, 0.f, .1f, .05f) - terrain) < 1e-3f);
        assert(depth_of(ground, .02f, .1f, .05f) > terrain);              // a leg above ground is in front
    }
}
''')


    def test_shadow_lies_on_the_terrain_at_any_height(self):
        # The projected shadow used 1.75 * relief per unit of ground height, far
        # below the terrain's 0.0016 * view height, so on any raised ground it lay
        # beneath the terrain and animated animals had no visible shadow.
        shader = (ROOT / "Renderer/lab/shared/shaders/objects/resource_skinning.hlsl").read_text()
        entry = shader[shader.index("VSResourceShadow"):]
        start = entry.index("input.position=float3(sx,sy,") + len("input.position=float3(sx,sy,")
        key = entry[start:entry.index(");", start)]
        run_cpp(r"""
#include <cassert>
#include <initializer_list>
struct float4 { float x, y, z, w; };
float shadow_key(float ground, float sy) {
    float4 resource_offset{0, 0, 0, ground};
    float4 resource_projection{64.f, 32.f, 1.14f, 1500.f};
    return """ + key + r""";
}
int main() {
    for (float ground : {0.f, 5.f, 20.f, 60.f}) {
        float base = 412.f;                                   // the shadowed ground point's unraised screen y
        float sy = base - ground * 1.14f * .82f;              // drawn on the raised ground
        float terrain = base + ground * .0016f * 1500.f;      // natural terrain depth there
        float key = shadow_key(ground, sy);
        assert(key > terrain);                                // on top of the ground, not beneath it
        assert(key < terrain + .02f * 150.f / .82f * .0016f * 1500.f);   // behind a hoof .02 tile up
    }
}
""")


if __name__ == "__main__":
    unittest.main()
