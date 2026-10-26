"""Ground that lies on the sea does not enter the sea's mirror.

Forest, jungle and surface (dune, plains) floor decals ride .18 height units
over their ground, and only their opaque core keeps off the shore. A rim that
reached past the optical shore floated just above the water plane, where the
game's mirror kept ground more than .0001 world units up. Its mirror image
was a faint premultiplied colour on the same spot; the coastal sea counts any
mirrored ground colour as full coverage, so the shallows showed dark, blocky
blotches beside coastal forests, strongest away from the screen centre where
the view ray grazes (user report, October 21; reproduced on the 3350 BC save).

The mirror now keeps only ground more than one height unit above the plane.
Ground below that would mirror onto itself, less than a pixel away.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / 'Renderer/sandbox/fresh_pipeline.h'
REFLECTION = ROOT / 'Renderer/native/environment_refresh/reflection.h'
DECALS = (ROOT / 'Renderer/lab/shared/natural/vegetation_floor_mesh_body.h',
          ROOT / 'Renderer/lab/shared/natural/surface_mesh_body.h')
GROUND = ROOT / 'Renderer/lab/shared/natural/ground.h'
RENDERER = ROOT / 'Renderer/native/c3x_renderer.cpp'
# Lab previews mirror terrain and mountains through their own PSReflection
# (environment_refresh/prepare.py, regenerated into the city profile's shaders).
LAB_MIRRORS = tuple(ROOT / f'Renderer/native/{folder}/{name}.hlsl'
                    for folder in ('environment_refresh', 'city_fidelity') for name in ('terrain', 'mountain'))


def number(expression):
    """A literal such as .0001, 1./112 or 2.5f/112.f."""
    expression = expression.replace('f', '')
    assert re.fullmatch(r'[\d.]+(/[\d.]+)?', expression), expression
    value, _, divisor = expression.partition('/')
    return float(value) / (float(divisor) if divisor else 1)


def mirror_ground_clip():
    """World-z margin above the water plane below which mirrored ground is clipped."""
    source = PIPELINE.read_text()
    entry = source.index('Output PSSandboxReflectionTerrainMaterial(P input) {')
    body = source[entry:source.index('}', entry)]
    found = re.search(r'clip\(input\.world\.z-NativeReflection\.z-([\d./f]+)\);', body)
    assert found, 'the mirror ground clip changed; update this model'
    return number(found.group(1))


def lab_mirror_clips():
    """The Lab mirror's ground clip in each generated terrain and mountain shader."""
    clips = {}
    for path in LAB_MIRRORS:
        source = path.read_text()
        entry = source.index('float4 PSReflection(P input):SV_Target {')
        body = source[entry:source.index('}', entry)]
        found = re.search(r'clip\(input\.world\.z-NativeReflection\.z-([\d./f]+)\);', body)
        assert found, f'the Lab mirror clip changed in {path.name}; update this model'
        clips[path] = number(found.group(1))
    return clips


def game_terrain_mirror_coverage():
    """Least main-pass coverage the game's terrain mirror material keeps."""
    source = PIPELINE.read_text()
    entry = source.index('Output PSSandboxReflectionTerrainMaterial(P input) {')
    body = source[entry:source.index('}', entry)]
    found = re.search(r'clip\(output\.color\.a-([\d.]+)\);\n\s*float inverse=rcp\(output\.color\.a\);\n\s*'
                      r'output\.color\*=inverse;output\.normal\*=inverse;output\.world\*=inverse;'
                      r'output\.properties\*=inverse;', body)
    assert found, 'the game terrain mirror keeps faint ground'
    return number(found.group(1))


def lab_terrain_mirror_coverage():
    """Least main-pass coverage the Lab's terrain mirror keeps, per generated shader."""
    kept = {}
    for path in LAB_MIRRORS:
        if not path.name.startswith('terrain'):
            continue
        source = path.read_text()
        entry = source.index('float4 PSReflection(P input):SV_Target {')
        body = source[entry:source.index('}', entry)]
        found = re.search(r'clip\(color\.a-([\d.]+)\);\n return float4\(color\.rgb/color\.a,1\);', body)
        assert found, f'the Lab terrain mirror keeps faint ground in {path.name}'
        kept[path] = number(found.group(1))
    return kept


def decal_lifts():
    """Height units each terrain-decal family rides above its ground."""
    lifts = []
    for path in DECALS:
        text = path.read_text()
        lifts += [float(v) for v in re.findall(r'height_natural\([^;]*?\)\+([\d.]+)f', text)]
        # Desert dunes sit at a fixed height over the flat datum.
        lifts += [float(v) - 2.5 for v in
                  re.findall(r'project_natural\(world_x,world_y,biome==2\?([\d.]+)f:', text)]
    return lifts


class MirrorGroundClipTests(unittest.TestCase):
    def test_conventions(self):
        # Height units become world z through /112; the mirror plane is the
        # 2.5-unit datum, which flat beach and sea ground share.
        self.assertIn('out.world_z=height/112;', GROUND.read_text())
        self.assertIn('{height_pixels,depth_metric,2.5f/112.f,', REFLECTION.read_text())
        self.assertIn('reflection.height_pixels=112.f*.82f*float(frame.tile_width)/224.f;',
                      RENDERER.read_text())
        self.assertGreaterEqual(len(decal_lifts()), 3)

    def test_decal_rims_over_the_sea_stay_out_of_the_mirror(self):
        clip = mirror_ground_clip()
        plane = 2.5 / 112
        for lift in decal_lifts():
            # A decal vertex over the sea: ground on the datum plus its lift.
            z = (2.5 + lift) / 112
            self.assertLess(z - plane - clip, 0, f'a {lift}-unit decal rim over the sea is mirrored')

    def test_only_ground_that_mirrors_onto_itself_is_clipped(self):
        # Clipped ground's mirror image lands 2 * height * height_pixels below
        # it. At gameplay zoom (128-pixel tiles) that stays under a pixel, so
        # coastal land and cliffs still reflect.
        height_pixels = 112 * .82 * 128 / 224
        self.assertLessEqual(2 * mirror_ground_clip() * height_pixels, 1.0)

    def test_the_lab_mirror_clips_as_the_game_does(self):
        # The Lab kept the old margin, so its previews still showed the
        # blotches (user report, October 25).
        for path, clip in lab_mirror_clips().items():
            self.assertEqual(clip, mirror_ground_clip(), path.relative_to(ROOT))

    def test_the_lab_sea_mirrors_no_faint_coastal_ground(self):
        # Mirrored terrain writes color only (the ground blend keeps no alpha)
        # and the sea takes any mirrored color as full coverage. The land's
        # coastal margin fades to a few percent over the beach, so over open
        # sea its near-black mirror image read as solid: dark patches offshore
        # (user report, October 25). Only ground drawn at least half opaque
        # mirrors, at its own color.
        water = (ROOT / 'Renderer/lab/shared/shaders/hydrology/water_natural.hlsl').read_text()
        self.assertIn('step(1e-6,max(object.r,max(object.g,object.b)))', water)
        for path, kept in lab_terrain_mirror_coverage().items():
            self.assertGreaterEqual(kept, .5, path.relative_to(ROOT))

    def test_the_game_sea_mirrors_no_faint_coastal_ground(self):
        # The game's coastal sea has the same rule, and its mirror material
        # kept the faded margin too (the waterline residue left after the
        # October 21 fix).
        water = (ROOT / 'Renderer/sandbox/water_surface.hlsl').read_text()
        self.assertIn('step(1e-6, max(object.r, max(object.g, object.b)))', water)
        self.assertGreaterEqual(game_terrain_mirror_coverage(), .5)

    def test_the_old_clip_kept_decal_rims(self):
        # Confirms the check against the margin behind the October 21 report.
        plane = 2.5 / 112
        self.assertTrue(any((2.5 + lift) / 112 - plane - .0001 > 0 for lift in decal_lifts()))


if __name__ == '__main__':
    unittest.main()
