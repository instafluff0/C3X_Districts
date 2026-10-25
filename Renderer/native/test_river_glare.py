"""A river's sun glint stays a small, soft sheen wherever it is on screen.

The river material measures its view ray from the screen centre, so its sun
and moon glint sit in a zone fixed to the screen. The glint once used the open
sea's strength (.9) and a lobe about 35 degrees wide: every river crossing
the right half of the view at noon turned pale white, its ripple normals
showing as swirls and its stream lines lost (user report, October 17).
Scrolling the same river to the left of centre made it blue again. The glint
now has a lobe about 18 degrees wide at .35 strength.

This test reads the glint terms from the source and every generated shader
(and the game pack, when present) and evaluates the glint of a calm river
over the Lab's 1024x768 view at gameplay zoom, through the day.
"""
import math
import re
import unittest

from Renderer.lab.platform import ROOT

SOURCE = 'Renderer/lab/shared/shaders/hydrology/scene_material_v1.hlsl'
GENERATED = (
    'Renderer/native/source_fidelity/generated/shaders/hydrology/scene_material_v1.hlsl',
    'Renderer/native/source_fidelity/hydrology.hlsl',
    'Renderer/native/environment_refresh/hydrology.hlsl',
    'Renderer/native/city_fidelity/hydrology.hlsl',
    'Renderer/native/city_fidelity/resource_shadow.hlsl',
)
PACK = 'Renderer/packs/Renderer64ResidentRuntime'
SUN = re.compile(r'float glint=smoothstep\(([.\d]+),([.\d]+),facing\)\*([.\d]+);')
MOON = re.compile(r'float moon_glint=smoothstep\(([.\d]+),([.\d]+),max\(dot\(reflect\(-environment_moon_direction,'
                  r'glint_normal\),eye\),0\)\)\s*\*smoothstep\(\.18,\.28,environment_moon_intensity\)\*([.\d]+);')
EYE = 'eye=normalize(float3(float2(.43,-.43)*2.5-delta,2.5));'
RUNTIME = 'Renderer/native/environment_runtime.cpp'
SHADOW_SLOPE = 1.35  # scene_lighting.h shadow_slope
PALE_LIMIT = .05
KEY = re.compile(r'\{([\d.]+)f, ([\d.]+)f, ([\d.]+)f, ([\d.]+)f, ([\d.]+)f\}')


def river_branch(text):
    begin = text.index('#ifdef Q3_CONTINUOUS_RIVERS\n  float2 world=q3_source_world(input);')
    return text[begin:text.index('#elif defined(Q3_STATIC_OPTICS_V2)', begin)]


def glints(text):
    branch = river_branch(text)
    assert EYE in branch, 'the river view ray changed; update this model'
    return tuple(float(v) for v in SUN.search(branch).groups()), tuple(float(v) for v in MOON.search(branch).groups())


def smoothstep(low, high, value):
    t = min(max((value - low) / (high - low), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def normalize(vector):
    length = math.sqrt(sum(v * v for v in vector))
    return [v / length for v in vector]


def direct_keys():
    """The sun's hour keys (hour, r, g, b, moon share) from environment_runtime.cpp."""
    text = (ROOT / RUNTIME).read_text()
    block = text[text.index('constexpr float direct_keys[][5]'):]
    keys = [tuple(float(v) for v in match.groups()) for match in KEY.finditer(block[:block.index('};')])]
    assert len(keys) >= 3, 'the direct key table changed; update this model'
    return keys


def sun(hour):
    """Direction toward the sun and its intensity: environment_runtime.cpp's
    sun keys and scene_lighting.h's key light (fixed slope), which every
    receiver gets except the legacy frozen profile."""
    phase = hour * math.pi / 12
    cast_x, cast_y = math.cos(phase), -math.sin(phase)
    world_x, world_y = cast_x + 2 * cast_y, cast_x - 2 * cast_y
    horizontal = math.hypot(world_x, world_y)
    keys = direct_keys()
    after = next((index for index, key in enumerate(keys) if key[0] > hour), len(keys))
    a, b = keys[after - 1], keys[after % len(keys)]
    t = smoothstep(0, 1, ((hour - a[0]) % 24) / ((b[0] - a[0]) % 24 or 24))
    intensity = max(a[channel] + (b[channel] - a[channel]) * t for channel in (1, 2, 3))
    return normalize([-world_x / horizontal, -world_y / horizontal, SHADOW_SLOPE]), intensity


def glint_field(low, high, strength, light, width=1024, height=768, tile=128, step=8):
    """A calm (flat) river's glint at each sampled view pixel."""
    reflected = [-light[0], -light[1], light[2]]
    values = []
    for sy in range(-height // 2, height // 2, step):
        for sx in range(-width // 2, width // 2, step):
            raw_x, raw_y = sx / (tile / 2), sy / (tile / 4)
            dx, dy = (raw_x + raw_y) / 2, (raw_x - raw_y) / 2
            eye = normalize([1.075 - dx, -1.075 - dy, 2.5])
            facing = max(0.0, sum(a * b for a, b in zip(reflected, eye)))
            values.append(smoothstep(low, high, facing) * strength)
    return values


def pale_share(glint):
    """Largest share of the view, over the day, where the glint adds more than
    .2 of sunlight to a river: enough to turn its dark water pale."""
    shares = []
    for hour in range(6, 19):
        light, intensity = sun(hour)
        field = glint_field(*glint, light)
        shares.append(sum(v * intensity > .2 for v in field) / len(field))
    return max(shares)


class RiverGlareTests(unittest.TestCase):
    def test_generated_shaders_and_pack_carry_the_source_glint(self):
        expected = glints((ROOT / SOURCE).read_text())
        for name in GENERATED:
            self.assertEqual(glints((ROOT / name).read_text()), expected, name)
        pack = ROOT / PACK
        if pack.is_dir():
            for name in GENERATED:
                self.assertEqual(glints((pack / name).read_text()), expected, PACK + '/' + name)

    def test_river_glint_is_a_small_soft_sheen(self):
        glint, moon = glints((ROOT / SOURCE).read_text())
        # The old wide, open-sea glint whitened about 8% of the view all day.
        self.assertLess(pale_share(glint), PALE_LIMIT)
        self.assertLessEqual(glint[2], .4, 'peak glint too strong for a river')
        # The moon shares the lobe and stays fainter than the sun.
        self.assertEqual(moon[:2], glint[:2])
        self.assertLessEqual(moon[2], glint[2])

    def test_the_old_glint_whitened_the_glare_zone(self):
        # Confirm the check against the glint the user reported.
        self.assertGreater(pale_share((.82, .995, .9)), PALE_LIMIT * 1.5)
        # Noon: right of centre, as the Lab render and the in-game report show.
        field = glint_field(.82, .995, .9, sun(12)[0])
        columns = 1024 // 8
        pale = [i % columns * 8 for i, v in enumerate(field) if v > .1]
        self.assertGreater(min(pale), 512)


if __name__ == '__main__':
    unittest.main()
