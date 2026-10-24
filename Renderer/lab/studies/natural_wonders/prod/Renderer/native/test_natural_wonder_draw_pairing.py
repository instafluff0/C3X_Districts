"""A natural-wonder body binds its Winter constants (C3XWonderBody, PS b13) only for its own
draw: every path that binds them unbinds them before the next chunk, with no early exit
between, or ordinary bodies would take a wonder's snow floor."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIND = re.compile(r"(?<!un)bind_wonder_body\(context")
UNBIND = re.compile(r"unbind_wonder_body\(context\)")
EXIT = re.compile(r"\b(continue|return|break|goto)\b")


def pairs(source: str) -> list[tuple[int, int]]:
    binds = [m.start() for m in BIND.finditer(source)]
    unbinds = [m.start() for m in UNBIND.finditer(source)]
    if len(binds) != len(unbinds):
        raise AssertionError(f"{len(binds)} binds but {len(unbinds)} unbinds")
    return list(zip(binds, unbinds))


class WonderBodyBindingTests(unittest.TestCase):
    def test_every_bind_is_unbound_on_the_same_path(self):
        for relative in ("Renderer/native/c3x_renderer.cpp", "Renderer/sandbox/fresh_pipeline.h"):
            source = (ROOT / relative).read_text()
            found = pairs(source)
            with self.subTest(file=relative):
                self.assertTrue(found, "no wonder-body draw binding")
                for index, (bind, unbind) in enumerate(found):
                    self.assertLess(bind, unbind)
                    if index + 1 < len(found):
                        self.assertLess(unbind, found[index + 1][0])
                    self.assertIsNone(EXIT.search(source[bind:unbind]), "an early exit skips the unbind")

    def test_bodies_take_the_feature_layer_projection(self):
        # Wonder bodies are placed by the same append_instance call as resource bodies, so in
        # world-objects mode (the game's resident world) their chunks must be tile-local world
        # chunks (projection kind 2) like every feature chunk. Cached as kind 0 they drew in
        # game with their low parts lost under the ground while their shadows remained.
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        feature = re.search(r"world_objects && layer>=geometry_route && layer<geometry_natural_terrain\?\s*"
                            r"\(layer>=geometry_cliff0\?4u:2u\)", source)
        self.assertIsNotNone(feature, "feature layers no longer cache as projection kind 2")
        start = source.index("auto& vertices=wonder_vertices[mask]")
        call = source[source.index("cache_geometry_layer(", start):source.index(")){", start)]
        self.assertIn("world_objects?2u:0u", call)

    def test_lake_and_channel_take_their_layers_projection(self):
        # A wonder's lake (water and bed) and fjord channel (river) are built with the
        # per-tile ground projection, like every ground layer chunk, which caches as kind 3
        # in world-ground mode (the game's resident world). Cached as kind 0 the channel drew
        # in game only past the relief, dark inside the fjord walls.
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        self.assertRegex(source, r"world_ground && layer<geometry_route\?3u:", "ground layers no longer cache as kind 3")
        for name in ("wonder_water", "wonder_river"):
            calls = [m.start() for m in re.finditer(r"cache_geometry_layer\(mesh_upload,%s," % name, source)]
            with self.subTest(chunk=name):
                self.assertTrue(calls, f"no {name} chunk")
                for start in calls:
                    self.assertIn("world_ground?3u:0u", source[start:source.index(")){", start)])

    def test_wonder_water_keeps_part_of_the_cast_shadow(self):
        # User, 2026-10-23: a wonder's lake and fjord channel read like the approved Lab look,
        # not a dark hole between their walls. Their vertices mark shape_visibility.x 2 (no
        # other water writes it), and the shared receiver query keeps a third of the cast
        # shadow for marked water only.
        shader = (ROOT / "Renderer/lab/shared/shaders/lighting/scene_shadow_v1.hlsl").read_text()
        start = shader.index("float q6_receiver_visibility(PixelInput input")
        body = shader[start:shader.index("\n}\n", start)]
        self.assertIn("if(water && input.shape_visibility.x>1.5)visibility=lerp(1.0,visibility,.35);", body)
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        block = source[source.index("std::vector<Vertex> wonder_water;"):source.index("if(!water_coverage && wonder_water.empty())")]
        self.assertIn("out.shadow_visibility=2;", block)

    def test_checker_rejects_an_early_exit(self):
        broken = "bind_wonder_body(context,1);if(x)continue;draw();unbind_wonder_body(context);"
        (bind, unbind), = pairs(broken)
        self.assertIsNotNone(EXIT.search(broken[bind:unbind]))
        with self.assertRaises(AssertionError):
            pairs("bind_wonder_body(context,1);draw();")


if __name__ == "__main__":
    unittest.main()
