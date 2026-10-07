"""Baked compositions can place animated subjects.

Land animals were anchored at the tile centre on the pickup relief alone, so on
hills they stood on the crown with heads sunk into the slope, and on raised
ground their legs disappeared. Compositions now name an animated subject as an
"animated/<binding>" asset (a placeholder mesh) with a baked position, facing
and size. The loader marks such compositions as animated, so the scheduler keeps
redrawing them, and the renderer turns each one into a resource anchor on the
natural surface. This test round-trips a bundle written by the composition
builder through the native loader.
"""
import struct
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.tools.asset_compiler.build_resource_compositions import MAGIC, mesh_payload
from Renderer.tools.asset_compiler.build_resource_runtime import bundle_string


def bundle() -> bytes:
    triangle = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0))] * 3
    rock = [((-.1, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0)), ((.1, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0)),
            ((0.0, .1, .1), (0.0, 0.0, 1.0), (0.0, 1.0))]
    blob = bytearray(MAGIC) + struct.pack("<IIII", 2, 1, 2, 0) + bundle_string("textures/atlas_0.dds")
    blob += mesh_payload("animated/horses", 0, triangle, [0, 1, 2]) + mesh_payload("source/rock", 0, rock, [0, 1, 2])
    blob += struct.pack("<I", 2)
    for name, assets in (("horses", (0, 1)), ("iron", (1,))):
        blob += bundle_string(name) + struct.pack("<III", 1, 1 << 2, len(assets))
        for asset in assets:
            blob += struct.pack("<I6f", asset, .5, .5, .3, 1.1, 0.0, .04)
    blob += struct.pack("<I", 2) + bundle_string("Horses") + struct.pack("<I", 0) + bundle_string("Iron") + \
        struct.pack("<I", 1)
    return bytes(blob)


class AnimatedCompositionTests(unittest.TestCase):
    def test_loader_marks_compositions_that_place_animated_subjects(self):
        data = ",".join(str(value) for value in bundle())
        run_cpp(r'''
#include "Renderer/native/terrain_scene_runtime.h"
#include <cassert>
#include <cmath>
#include <cstdio>
static unsigned char const data[] = {''' + data + r'''};
int main() {
    // The bundle is written beside the test executable on either platform.
    std::FILE * file = std::fopen("animated_composition.bin", "wb");
    assert(file && std::fwrite(data, 1, sizeof(data), file) == sizeof(data));
    std::fclose(file);
    c3x_renderer::FeatureBundle bundle;
    assert(c3x_renderer::load_feature_bundle("animated_composition.bin", bundle));
    auto const * horses = c3x_renderer::find_feature_composition(bundle, "Horses");
    auto const * iron = c3x_renderer::find_feature_composition(bundle, "Iron");
    assert(horses && iron && horses != iron);
    assert(horses->animated);
    assert(!iron->animated);
    auto const & subject = horses->variants[0].instances[0];
    assert(bundle.assets[subject.asset].id == "animated/horses");
    assert(std::abs(subject.rotation - .3f) < 1e-6f && std::abs(subject.scale - 1.1f) < 1e-6f);
    assert(std::abs(subject.ground_fit - .04f) < 1e-6f);
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))
    def test_runtime_loads_only_the_subjects_a_pack_names(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        start = source.index("std::vector<std::string> names = {")
        names = source[start:source.index("for (auto const & name_entry : names) {", start)]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
struct Asset { std::string id; };
struct Bundle { std::vector<Asset> assets; };
std::vector<std::string> requested(Bundle const & resource_bundle, bool resource_assets_ready) {
    ''' + names + r'''
    return names;
}
int main() {
    Bundle production{{{"resource/assets/res_iron_rock01"}, {"source/res_wheat_tuft01"}}};
    assert(requested(production, true).size() == 10);
    Bundle lab{{{"animated/cattle.3"}, {"animated/horses"}, {"animated/cattle.3"}, {"source/boulder_01"}}};
    auto names = requested(lab, true);
    assert(names.size() == 11 && names.back() == "cattle.3");     // added once; primaries not duplicated
    assert(requested(lab, false).size() == 10);                    // an unloaded pack names nothing
}
''')

    def test_posed_body_follows_the_ground_plane(self):
        # Animals bend down to graze. Standing a body on one height leaves a lowered
        # head sunk in rising ground; the packed pose is sheared onto the ground plane.
        run_cpp(r'''
#include "Renderer/native/render_core/resource_instances.h"
#include <cassert>
#include <cmath>
#include <vector>
int main() {
    std::vector<float> values(16 + 2 * 28, 0.f);
    for (unsigned bone = 0; bone < 2; ++bone) {               // identity, then a lowered "head" bone
        float* m = values.data() + 16 + bone * 28;
        m[0] = m[5] = m[10] = 1.f; m[12] = bone ? .3f : 0.f; m[14] = bone ? -.05f : 0.f;
    }
    auto skin = [&](unsigned bone, float x, float y, float z, float* out) {
        float const* m = values.data() + 16 + bone * 28;
        for (unsigned a = 0; a < 3; ++a) out[a] = x * m[a] + y * m[4 + a] + z * m[8 + a] + m[12 + a];
    };
    float flat[3], head[3];
    skin(1, .1f, .2f, 0.f, head);
    c3x_renderer::render_core::slope_resource_pose(2, .25f, -.5f, .01f, values.data());
    skin(1, .1f, .2f, 0.f, flat);
    // Height rises exactly by the plane at the posed point; x and y are unchanged.
    assert(std::abs(flat[0] - head[0]) < 1e-6f && std::abs(flat[1] - head[1]) < 1e-6f);
    assert(std::abs(flat[2] - (head[2] + .25f * head[0] - .5f * head[1] + .01f)) < 1e-6f);
    std::vector<float> level(values.size(), 0.f);
    c3x_renderer::render_core::slope_resource_pose(2, 0.f, 0.f, 0.f, level.data());
    for (float value : level) assert(value == 0.f);           // flat ground leaves the pose untouched
}
''')


if __name__ == "__main__":
    unittest.main()
