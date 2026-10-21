"""A baked resource composition wins over the animation pack's alias.

A resource without a composition is animated when the animation pack lists
its name ("aliases" in ResourceAnimationRuntime/bindings.json), matched exactly
and case-insensitively as compositions match theirs. It used to be a lowercase
substring of the resource name, so a scenario's "Wild Wheat" took the wheat
animation by accident. A pack that bakes a composition for a resource, such as
a static wheat field, must replace that presentation: otherwise the
composition branch was never reached and the Lab drew the single animated
subject instead. resource_animation_for, which both the build and the redraw
scheduler use, declines when a composition exists. This test compiles that
member function against a small shim.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def function(text, signature):
    start = text.index(signature)
    depth, at = 0, text.index('{', start)
    for index in range(at, len(text)):
        depth += {'{': 1, '}': -1}.get(text[index], 0)
        if depth == 0:
            return text[start:index + 1]
    raise ValueError(signature)


class ResourceCompositionPriorityTests(unittest.TestCase):
    def test_composition_beats_the_animation_alias(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        lookup = function(source, 'int resource_animation_for(c3x_renderer_tile_v1 const & tile) const')
        program = r'''
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <strings.h>
#define _stricmp strcasecmp
#endif
struct c3x_renderer_tile_v1 { int city_id; int resource_id; char resource_name[32]; };
namespace c3x_renderer {
struct FeatureComposition { std::string name; };
struct FeatureBundle { std::vector<FeatureComposition> compositions; };
FeatureComposition const * find_feature_composition(FeatureBundle const & bundle, char const * name) {
    for (auto const & item : bundle.compositions) if (item.name == name) return &item;
    return nullptr;
}
}
struct Animation { std::string name; };
struct Renderer {
    std::vector<Animation> resource_animations{{"banana"}, {"wheat"}};
    std::vector<std::pair<std::string, unsigned>> resource_animation_aliases{{"Bananas", 0}, {"Wheat", 1}};
    c3x_renderer::FeatureBundle resource_bundle;
    bool resource_assets_ready = true;
''' + lookup + r'''
};
int main() {
    Renderer renderer;
    c3x_renderer_tile_v1 tile{-1, 3, "Wheat"};
    int failures = 0;
    if (renderer.resource_animation_for(tile) != 1) { std::puts("FAIL animated match without composition"); ++failures; }
    renderer.resource_bundle.compositions.push_back({"Wheat"});
    if (renderer.resource_animation_for(tile) != -1) { std::puts("FAIL composition must win"); ++failures; }
    renderer.resource_assets_ready = false;
    if (renderer.resource_animation_for(tile) != 1) { std::puts("FAIL unloaded pack keeps animation"); ++failures; }
    std::strcpy(tile.resource_name, "Bananas");
    renderer.resource_assets_ready = true;
    if (renderer.resource_animation_for(tile) != 0) { std::puts("FAIL other names keep animation"); ++failures; }
    std::strcpy(tile.resource_name, "WHEAT");
    if (renderer.resource_animation_for(tile) != 1) { std::puts("FAIL aliases ignore case"); ++failures; }
    std::strcpy(tile.resource_name, "Wild Wheat");
    if (renderer.resource_animation_for(tile) != -1) { std::puts("FAIL no substring match"); ++failures; }
    std::strcpy(tile.resource_name, "banana");
    if (renderer.resource_animation_for(tile) != -1) { std::puts("FAIL binding names are not aliases"); ++failures; }
    return failures;
}
'''
        run_cpp(program)   # raises if any expectation fails


if __name__ == '__main__':
    unittest.main()
