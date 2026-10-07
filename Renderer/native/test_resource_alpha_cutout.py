"""Masked resource models cut out by their texture alpha.

Civ VI's wheat, wine, incense and similar crops are alpha-masked cards. The
static feature pass only clipped resource materials that carried a fractional
code (decals, mines, tile objects), so plain resource bodies (materials 21-28
with no fraction) ignored their slot alpha and drew every card as a solid
rectangle. The composition pipeline now merges each card's opacity into a BC3
atlas, and the feature shader clips resource bodies on that alpha too. This
test compiles the shader's material-weight and clip statements as C++. A
transparent texel of a resource body must be discarded and an opaque one kept,
opaque BC1 bodies (alpha 1) stay whole, decals keep their soft cutoff, and unit
materials are never clipped by this statement.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


class ResourceAlphaCutoutTests(unittest.TestCase):
    def test_resource_bodies_clip_transparent_texels(self):
        shader = (ROOT / 'Renderer/native/render_core/terrain_scene.hlsl').read_text()
        weights = between(shader, '    float material_fraction = frac(input.material_index);',
                          '    float mine_slot')
        statement = between(shader, '    clip(lerp(1.0, mine_sample.a', ';') + ';'
        program = r'''
#include <cstdio>
#include <cmath>
struct Input { float material_index; };
struct Sample { float a; };
static bool clipped;
float step(float edge, float x) { return x >= edge ? 1.0f : 0.0f; }
float lerp(float a, float b, float t) { return a + (b - a) * t; }
float saturate(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
float frac(float x) { return x - std::floor(x); }
void clip(float x) { if (x < 0) clipped = true; }
bool kept(float material, float alpha) {
    Input input{material}; Sample mine_sample{alpha}; clipped = false;
''' + weights + statement + r'''
    return !clipped;
}
int main() {
    int failures = 0;
    auto expect = [&](bool value, const char* what) { if (!value) { std::printf("FAIL %s\n", what); ++failures; } };
    expect(!kept(21.0f, 0.0f), "transparent texel of a resource body is discarded");
    expect(kept(21.0f, 1.0f), "opaque resource body texel is kept");
    expect(!kept(28.0f, 0.05f), "every resource slot clips below the body cutoff");
    expect(kept(24.0f, 0.5f), "half-opaque card texel is kept");
    expect(kept(21.35f, 0.01f), "soft decal texel above the decal cutoff is kept");
    expect(!kept(21.35f, 0.001f), "decal texel below the decal cutoff is discarded");
    expect(kept(21.42f, 0.0f), "unit materials are not clipped here");
    expect(kept(12.0f, 0.0f), "non-resource features are not clipped here");
    return failures;
}
'''
        run_cpp(program)   # raises if any expectation fails


if __name__ == '__main__':
    unittest.main()
