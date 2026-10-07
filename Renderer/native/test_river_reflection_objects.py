"""Rivers reflect standing objects and mountains, never the ground around them.

The mirror target reflects about the sea plane. Most land, river corridors
included, sits a little above that plane, so the mirrored ground landed on the
rivers beside it: long stretches of river showed grass and hills instead of
water. Mirror passes now draw ground (relief, terrain decals, mountains and
farm fields) with a color-only blend, so the mirror's alpha counts only the
standing objects drawn after it (forests, jungles, units and buildings) and
mountains: the user liked a snowy peak mirrored in a pool. The game relights
all mirrored terrain in one color-only pass, then redraws mountains with an
alpha-only blend. The river weighs the mirror by that alpha. The sea also
counts mirrored ground color as coverage, so coastal land still reflects.

This test reads the blend setup and ground layer lists from the sources and
composes one mirror pixel the way D3D11 blends it.
"""
import re
import unittest

from Renderer.lab.platform import ROOT


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def compose(mask_alpha, layers):
    """Premultiplied ONE / INV_SRC_ALPHA blending onto a cleared mirror texel."""
    rgb, alpha = 0.0, 0.0
    for color, coverage, ground in layers:
        if ground != 'alpha':  # an alpha-only pass leaves the color
            rgb = color * coverage + rgb * (1 - coverage)
        if not ground or ground == 'alpha' or mask_alpha:
            alpha = coverage + alpha * (1 - coverage)
    return rgb, alpha


class RiverReflectionObjectTests(unittest.TestCase):
    def ground_mask_writes_alpha(self, renderer):
        creation = between(renderer, 'hr = device->CreateBlendState(&blend, &blend_state);',
                           'hr = device->CreateBlendState(&blend, &reflection_terrain_blend);')
        mask = re.search(r'RenderTargetWriteMask = ([^;]+);', creation)[1]
        return 'ALPHA' in mask or 'ENABLE_ALL' in mask

    def test_rivers_reflect_objects_and_the_sea_keeps_terrain(self):
        renderer = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        # Lab and game mirror passes give ground layers the color-only state.
        lab = between(between(renderer, 'if(reflection_pass){', 'return draw_cached_geometry('),
                      'bool ground=', 'nullptr,0xffffffffu);')
        game = between(between(fresh, 'auto& mirror=sandbox_active_reflection();',
                               'if (layer>=geometry_natural_terrain)'),
                       'else if(mirrored && renderer.reflection_terrain_blend', 'nullptr,0xffffffffu);')
        for block in (lab, game):
            self.assertIn('reflection_terrain_blend', block)
            for layer in ('geometry_farm', 'geometry_natural_terrain', 'geometry_natural_decal'):
                self.assertIn(layer, block)
            for layer in ('geometry_feature', 'geometry_city', 'geometry_natural_forest0',
                          'geometry_natural_mountain'):
                self.assertNotIn(layer, block)
        # The cached-terrain mirror redraws mountains with alpha-only coverage.
        cached = between(fresh, 'if(cached_terrain){', '}else for(auto layer:')
        self.assertIn('draw(geometry_natural_mountain)', cached)
        self.assertIn('mirror_mountain_coverage=true;', cached)
        self.assertIn('reflection_coverage_blend', between(
            fresh, 'auto& mirror=sandbox_active_reflection();', 'else if(mirrored && renderer.reflection_terrain_blend'))
        coverage = between(renderer, 'hr = device->CreateBlendState(&blend, &reflection_terrain_blend);',
                           'hr = device->CreateBlendState(&blend, &reflection_coverage_blend);')
        self.assertIn('D3D11_COLOR_WRITE_ENABLE_ALPHA;', coverage)
        self.assertIn('reflection_terrain_blend', between(fresh, 'void relight_reflected_terrain()',
                                                          'context->Draw(3,0);'))
        writes_alpha = self.ground_mask_writes_alpha(renderer)
        # A river texel over grass with nothing standing on it, then one with a tree.
        grass = [(0.30, 1.0, True)]
        tree = grass + [(0.08, 0.9, False)]
        # A mountain: relit color first, then its alpha-only coverage pass.
        mountain = [(0.55, 1.0, True), (0.0, 1.0, 'alpha')]
        _, ground_alpha = compose(writes_alpha, grass)
        tree_rgb, tree_alpha = compose(writes_alpha, tree)
        peak_rgb, peak_alpha = compose(writes_alpha, mountain)
        self.assertEqual(ground_alpha, 0.0, 'mirrored ground gives rivers reflection coverage')
        self.assertAlmostEqual(tree_alpha, 0.9)
        self.assertEqual((peak_rgb, peak_alpha), (0.55, 1.0))
        # The river weighs the mirror by its alpha; the seas add ground presence.
        material = (ROOT / 'Renderer/lab/shared/shaders/hydrology/scene_material_v1.hlsl').read_text()
        self.assertIn('mirrored_alpha=saturate(mirrored.a)*inside', material)  # alpha only, no ground presence
        for path, needle in (('Renderer/sandbox/water_surface.hlsl',
                              'object_alpha = saturate(max(object.a, terrain_present))'),
                             ('Renderer/lab/shared/shaders/hydrology/water_natural.hlsl',
                              'object_coverage=max(object_coverage,step(1e-6,max(object.r')):
            self.assertIn(needle, (ROOT / path).read_text())
        grass_rgb, _ = compose(writes_alpha, grass)
        self.assertGreater(grass_rgb, 1e-6)  # the sea still sees the ground color

    def test_a_shaded_mountain_does_not_mirror_black(self):
        # The 1498 capture's pool under a mountain's shaded face (linear
        # luminance): the face, the sky the river reflects, the water body,
        # and the pool's Fresnel. The pool must read close to open sky water;
        # the old mix showed a near-black hole there.
        material = (ROOT / 'Renderer/lab/shared/shaders/hydrology/scene_material_v1.hlsl').read_text()
        mix = re.search(r'reflected=lerp\(reflected,mirrored\.rgb,([^;]+)\);', material)[1]
        weight = re.search(r'float reflection=([^;]+);', material)[1]
        face, lit, sky, body, fresnel = .065, .39, .85, .04, .45

        def pool(mirror, alpha):
            scope = {'lerp': lambda a, b, t: a + (b - a) * t, 'max': max,
                     'fresnel': fresnel, 'land_bank': 1.0, 'mirrored_alpha': alpha}
            reflected = sky + (mirror - sky) * eval(mix, scope)
            reflection = eval(weight, scope)
            return body + (reflected - body) * reflection

        open_sky = pool(sky, 0.0)
        self.assertGreater(pool(face, 1.0), .7 * open_sky, 'a shaded face mirrors near-black')
        self.assertGreater(pool(lit, 1.0), 1.3 * open_sky, 'a lit face no longer reads in the water')

    def test_the_old_full_mask_reflected_ground_in_rivers(self):
        # Confirm the check: drawing ground with alpha writes gives coverage 1.
        _, alpha = compose(True, [(0.30, 1.0, True)])
        self.assertEqual(alpha, 1.0)


if __name__ == '__main__':
    unittest.main()
