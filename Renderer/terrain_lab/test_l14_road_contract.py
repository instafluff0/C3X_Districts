import csv
import hashlib
import unittest
from collections import Counter, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = Path(__file__).with_name("fixtures") / "l14_roads_192.csv"
CPP = Path(__file__).with_name("terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = Path(__file__).with_name("terrain_lab.hlsl").read_text(encoding="utf-8")
RUN = Path(__file__).with_name("RUN_L14.bat").read_text(encoding="utf-8")


class L14RoadContractTests(unittest.TestCase):
    def test_lab_augmentation_is_dense_connected_and_explicit(self):
        rows = list(csv.reader(SCENARIO.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(rows[0][0], "C3X_LAB_ROAD_SCENARIO_V0")
        self.assertEqual(rows[0][5], "lab_augmentation")
        self.assertEqual(int(rows[0][1]) * int(rows[0][2]), 192)
        self.assertEqual(int(rows[0][3]), len(rows) - 1)
        adjacency = {}
        styles = set()
        wraps = bridges = pillaged = 0
        for row in rows[1:]:
            a, b = (int(row[0]), int(row[1])), (int(row[2]), int(row[3]))
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
            wraps += int(row[4])
            styles.add(int(row[5]))
            pillaged += int(row[6])
            bridges += int(row[7])
        reached = set()
        queue = deque([next(iter(adjacency))])
        while queue:
            node = queue.popleft()
            if node in reached:
                continue
            reached.add(node)
            queue.extend(adjacency[node] - reached)
        degrees = Counter({node: len(neighbors) for node, neighbors in adjacency.items()})
        self.assertEqual(reached, set(adjacency))
        self.assertGreaterEqual(sum(value >= 3 for value in degrees.values()), 20)
        self.assertGreaterEqual(sum(value == 1 for value in degrees.values()), 8)
        self.assertGreaterEqual(wraps, 2)
        self.assertGreaterEqual(bridges, 4)
        self.assertGreaterEqual(pillaged, 1)
        self.assertEqual(styles, {0, 1, 2, 3})

    def test_renderer_consumes_source_route_and_bridge_art(self):
        self.assertIn("load_road_scenario", CPP)
        self.assertIn("add_road_scene", CPP)
        self.assertIn("add_road_bridge_scene", CPP)
        self.assertIn("bridge_runtime.bin", CPP)
        self.assertIn("surface_kind > 10.5", HLSL)
        self.assertIn("road_base_texture_7", HLSL)
        self.assertIn("road_bridge_base_texture_5", HLSL)
        self.assertIn("continuous_ribbon", HLSL)
        self.assertIn("road_wave", CPP)
        self.assertIn("Civ5EnvironmentSkin", RUN)
        self.assertIn("Civ5EnvironmentVegetation", RUN)
        self.assertNotIn("railroad", RUN.lower())

    def test_checked_in_fixture_hash_is_stable(self):
        self.assertEqual(
            hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
            "e9bef85d65543e30bf7c1caade36cdd5b69ed097f984d8ddb75148bee70e88e2",
        )


if __name__ == "__main__":
    unittest.main()
