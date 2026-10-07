"""Lab city recomposition: denser bodies never collide, stay within bounds and
encode a valid version-five library; the production pack is never written."""
import json
import math
import unittest
from pathlib import Path

from Renderer.lab.studies.city_readability import recompose as rc
from Renderer.lab.studies.city_readability import runtime_pack as rp

SOURCE = rc.SOURCE / "city.bin"


@unittest.skipUnless(SOURCE.is_file(), "production city pack is local data")
class RecomposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = rp.decode(SOURCE)
        cls.records = json.loads((rc.SOURCE / "manifest.json").read_text())["models"]
        cls.style = json.loads(rc.STYLE.read_text())

    def composer(self):
        # Accents need the local accent pack; geometry rules hold without them.
        style = dict(self.style, accents={})
        return rc.Composer(self.library, self.records, style, {})

    def sample(self):
        seen = set()
        for index, t in enumerate(self.library["templates"]):
            key = (t["culture"], t["era"], t["size"], t["walled"])
            if t["variant"] == 0 and not t["capital"] and key not in seen:
                seen.add(key)
                yield index, t

    def test_codec_round_trips_the_production_library(self):
        self.assertEqual(rp.encode(self.library, 4), SOURCE.read_bytes())

    def test_recomposed_bodies_are_larger_denser_and_never_collide(self):
        composer = self.composer()
        grown = added = 0
        for index, template in self.sample():
            before = composer.items(template)
            instances, paving, stats = composer.compose(
                json.loads(json.dumps(template)), 0, ("test", index))
            limit = self.style["extent"][template["size"]]
            items = [rc.Item(i["model"], i["scale"], i["rotation"], i["offset"], i["lights"],
                             i["capital"], "") for i in instances]
            kinds = [rc.instance_kind(i, self.records) for i in instances]
            bodies = [composer.poly(item) for item, kind in zip(items, kinds) if kind != "tree"]
            original = [composer.poly(i) for i in before if i.kind != "tree"]
            # Bodies that already met in the authored layout may still meet.
            def authored_overlap(a, b):
                return any(rc.overlap(a, o, 0.0) for o in original) and any(rc.overlap(b, o, 0.0) for o in original)
            for i, a in enumerate(bodies):
                for b in bodies[i + 1:]:
                    if rc.overlap(a, b, 0.0):
                        self.assertTrue(authored_overlap(a, b), f"template {index}: new collision")
            for item, kind, poly in zip(items, kinds, [composer.poly(i) for i in items]):
                if kind in ("building", "accent") and item.scale > 0:
                    self.assertLessEqual(rc.reach(poly), max(limit, 1.0) + 1e-6)
            self.assertLessEqual(len(instances), 128)
            self.assertLessEqual(sum(len(i["lights"]) for i in instances), 128)
            self.assertLessEqual(len(paving["vertices"]), 30000)
            self.assertTrue(all(0 <= v[2] <= 1 for v in paving["vertices"]))
            self.assertTrue(all(i < len(paving["vertices"]) for i in paving["indices"]))
            # Ordinary buildings may yield to the site; the palace, walls and
            # at most one civic core at the tile centre always stay.
            fixed = [i for i, kind in zip(instances, kinds)
                     if kind == "building" and not i["flags"] & rp.SITE_OPTIONAL]
            self.assertLessEqual(len(fixed), 1)
            if fixed:
                self.assertLess(math.hypot(*fixed[0]["offset"]), .14)
            for instance, kind in zip(instances, kinds):
                if kind in ("palace", "wall"):
                    self.assertFalse(instance["flags"] & rp.SITE_OPTIONAL)
            grown += sum(1 for i in instances if i["flags"] & rp.SITE_OPTIONAL) > 0
            added += stats["infill"]
        self.assertGreater(added, 0)

    def test_growth_scales_bodies_uniformly_and_moves_their_lights(self):
        composer = self.composer()
        item = rc.Item(0, 2.0, 0.0, (0.0, 0.0), [[.1, .2, .3, .2, 1, 1, 1, 1, 1, 0, 0, 0]], 0, "building")
        items = [item]
        composer.grow(items, 1.5, 10.0)
        self.assertAlmostEqual(item.scale, 3.0)
        light = item.lights[0]
        self.assertAlmostEqual(light[0], .15)
        self.assertAlmostEqual(light[1], .3)
        self.assertAlmostEqual(light[2], .45)

    def test_version_five_encoding_carries_flags_and_look(self):
        library = json.loads(json.dumps({k: v for k, v in self.library.items() if k != "models"}))
        library["models"] = self.library["models"]
        library["templates"] = library["templates"][:1]
        library["templates"][0]["instances"][0]["flags"] = rp.SITE_OPTIONAL | rp.TREE
        library["look"] = [.5, .25, 0, 0, 0, 0, 0, 0]
        data = rp.encode(library, 5)
        path = Path(__file__).resolve().parents[2] / "out" / "city-study" / "test-v5.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        try:
            decoded = rp.decode(path)
        finally:
            path.unlink()
        self.assertEqual(decoded["version"], 5)
        self.assertEqual(decoded["look"][:2], [.5, .25])
        self.assertEqual(decoded["templates"][0]["instances"][0]["flags"], rp.SITE_OPTIONAL | rp.TREE)

    def test_never_targets_the_production_pack(self):
        with self.assertRaises(ValueError):
            rc.build(rc.SOURCE)


if __name__ == "__main__":
    unittest.main()
