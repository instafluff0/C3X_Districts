"""Resource bodies take a readability look from their pack, as units do.

Civ VI resource bodies are dark (the elephant's albedo is sRGB ~71) and most of
their visible faces turn from the key light, where the shared feature response
gives only ambient plus 16% of the sun. In game the filmic toe then pressed the
elephant to luma ~40 beside grass at ~170 (Civ III's sprite: ~122). Units fixed
the same problem with a look (unit_quality.json: extra saturation, extra gain,
a highlight shoulder). Resource bodies now take that look from their pack's
look.json (profiles "look"), through the terrain settings constant buffer.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

NATIVE = ROOT / "Renderer/native"


def hlsl_offsets(block):
    """HLSL constant-buffer packing: a member never straddles a 16-byte row."""
    sizes = {"float": 4, "float2": 8, "float3": 12, "float4": 16}
    offset, result = 0, {}
    for kind, name in re.findall(r"^\s*(float[234]?)\s+(\w+);", block, re.MULTILINE):
        size = sizes[kind]
        if offset // 16 != (offset + size - 1) // 16:
            offset = (offset + 15) // 16 * 16
        result[name] = offset
        offset += size
    return result


class ResourceLookTests(unittest.TestCase):
    def test_cpu_settings_match_the_game_constant_buffer(self):
        source = (NATIVE / "terrain_rendering.hlsl").read_text()
        game = source[source.index("#ifdef C3X_GAME_RENDERER"):source.index("#define lab_mode")]
        offsets = hlsl_offsets(game[game.index("{"):game.index("};")])
        renderer = (NATIVE / "c3x_renderer.cpp").read_text()
        struct = renderer[renderer.index("struct TerrainShaderSettings {"):]
        struct = struct[:struct.index("};") + 2]
        run_cpp("#include <cassert>\n#include <cstddef>\n" + struct + r'''
int main() {
    static_assert(offsetof(TerrainShaderSettings, hour) == %d, "hour");
    static_assert(offsetof(TerrainShaderSettings, resource_look) == %d, "look follows the HLSL row");
    static_assert(sizeof(TerrainShaderSettings) %% 16 == 0, "constant buffers are whole rows");
}
''' % (offsets["game_hour"], offsets["game_resource_look"]))

    def test_generated_feature_shaders_apply_the_look_to_bodies_only(self):
        generator = (NATIVE / "render_core/generate_shaders.py").read_text()
        self.assertIn("if (resource_weight > 0.5 && (game_resource_look.x > 0 || game_resource_look.y > 0))",
                      generator)
        # The same response as the units' look (prepare_unit_shader.py).
        unit = (NATIVE / "environment_refresh/prepare_unit_shader.py").read_text()
        self.assertIn("if(peak>1.6)radiance*=(1.6+(peak-1.6)/(1+(peak-1.6)/.8))/peak;", unit)
        self.assertIn("lit_color *= (1.6 + (look_peak - 1.6) / (1 + (look_peak - 1.6) / 0.8)) / look_peak;",
                      generator)
        for name in ("feature.hlsl", "rigid_feature.hlsl", "resource_body.hlsl"):
            shader = (NATIVE / "city_fidelity" / name).read_text()
            self.assertTrue("float4 game_resource_look;" in shader, name)
            self.assertEqual(shader.count("(1 + game_resource_look.x);"), 1, name)

    def test_pack_look_reaches_both_settings_writers(self):
        from Renderer.tools.asset_compiler import build_resource_compositions as compositions
        profiles = json.loads(compositions.PROFILES.read_text())
        self.assertEqual(set(profiles["look"]) - {"note"}, {"gain", "saturation"})
        builder = (ROOT / "Renderer/tools/asset_compiler/build_resource_compositions.py").read_text()
        self.assertIn('(output / "look.json").write_text(', builder)
        renderer = (NATIVE / "c3x_renderer.cpp").read_text()
        self.assertIn('read_file((resource_root + "\\\\look.json").c_str(), look_data)', renderer)
        self.assertIn('json_number_after(look_data, "look_gain", 0, resource_look[0]);', renderer)
        self.assertIn('json_number_after(look_data, "look_saturation", 0, resource_look[1]);', renderer)
        self.assertIn('"C3X_RENDERER_RESOURCE_LOOK"', renderer)
        self.assertIn("std::begin(frame_settings.resource_look)", renderer)
        fresh = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        self.assertIn("std::copy(std::begin(renderer.resource_look),std::end(renderer.resource_look),"
                      "settings.resource_look);", fresh)


if __name__ == "__main__":
    unittest.main()
