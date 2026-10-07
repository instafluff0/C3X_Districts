"""The city overlay replaces only the city material region of a pack shader."""
import unittest

from Renderer.tools.overlay_city_shading import END, START, city_overlay

PACK = ("// farm overlay A\n" + START + " { float4 CityMaterialFlags; float4 CityAtlas; };\n"
        "body old\n" + END + "\nVS old\n// route overlay B\n")
SOURCE = ("// farm unaccepted\n" + START + " { float4 CityMaterialFlags; float4 CityAtlas; float4 CityLight; };\n"
          "#define Q8_CITY_TIME CityLight.w\nbody new\n" + END + "\nVS new\n// route unaccepted\n")


class CityOverlay(unittest.TestCase):
    def test_replaces_only_the_city_region(self):
        result, applied = city_overlay(PACK, SOURCE)
        self.assertTrue(applied)
        self.assertIn("float4 CityLight;", result)
        self.assertIn("body new", result)
        self.assertNotIn("body old", result)
        # Other features' code on either side stays the pack's.
        self.assertTrue(result.startswith("// farm overlay A\n"))
        self.assertTrue(result.endswith(END + "\nVS old\n// route overlay B\n"))
        self.assertNotIn("unaccepted", result)

    def test_requires_the_city_change_in_the_source(self):
        stale = SOURCE.replace("#define Q8_CITY_TIME CityLight.w\n", "")
        self.assertEqual(city_overlay(PACK, stale), (PACK, False))

    def test_idempotent_and_marker_bounded(self):
        once, _ = city_overlay(PACK, SOURCE)
        self.assertEqual(city_overlay(once, SOURCE), (once, False))
        self.assertEqual(city_overlay(PACK.replace(END, ""), SOURCE), (PACK.replace(END, ""), False))


if __name__ == "__main__":
    unittest.main()
