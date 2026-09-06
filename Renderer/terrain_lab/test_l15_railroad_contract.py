import csv
import hashlib
import unittest
from collections import Counter, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = Path(__file__).with_name("fixtures") / "l15_railroads_192.csv"
CPP = Path(__file__).with_name("terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = Path(__file__).with_name("terrain_lab.hlsl").read_text(encoding="utf-8")
RUN = Path(__file__).with_name("RUN_L15.bat").read_text(encoding="utf-8")


class L15RailroadContractTests(unittest.TestCase):
    def test_lab_augmentation_is_connected_dense_and_wrap_safe(self):
        rows = list(csv.reader(SCENARIO.read_text(encoding="utf-8").splitlines()))
        self.assertEqual("C3X_LAB_RAILROAD_SCENARIO_V0", rows[0][0])
        self.assertEqual("lab_augmentation", rows[0][5])
        self.assertEqual(192, int(rows[0][1]) * int(rows[0][2]))
        self.assertEqual(int(rows[0][3]), len(rows) - 1)
        adjacency = {}
        wraps = bridges = pillaged = 0
        for row in rows[1:]:
            a, b = (int(row[0]), int(row[1])), (int(row[2]), int(row[3]))
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
            wraps += int(row[4])
            bridges += int(row[7])
            pillaged += int(row[6])
            self.assertEqual(4, int(row[5]))
        reached = set()
        queue = deque([next(iter(adjacency))])
        while queue:
            node = queue.popleft()
            if node not in reached:
                reached.add(node)
                queue.extend(adjacency[node] - reached)
        degrees = Counter({node: len(neighbors) for node, neighbors in adjacency.items()})
        self.assertEqual(set(adjacency), reached)
        self.assertGreaterEqual(len(rows) - 1, 50)
        self.assertGreaterEqual(sum(value >= 3 for value in degrees.values()), 8)
        self.assertGreaterEqual(wraps, 2)
        self.assertGreaterEqual(bridges, 4)
        self.assertGreaterEqual(pillaged, 1)

    def test_renderer_uses_authored_rail_ballast_sleeper_and_bridge_sources(self):
        self.assertIn("add_railroad_scene", CPP)
        self.assertIn("add_railroad_bridge_scene", CPP)
        self.assertIn("sample_railroad_source", HLSL)
        self.assertIn("railroad_base_texture_0", HLSL)
        self.assertIn("railroad_base_texture_1", HLSL)
        self.assertIn('railroad ? "railroad"', CPP)
        self.assertIn("Civ5EnvironmentSkin", RUN)
        self.assertIn("l15_railroads_192.csv", RUN)
        self.assertNotIn("train", RUN.lower())

    def test_checked_in_fixture_hash_is_stable(self):
        self.assertEqual(
            "8cbeccbb1806425654cf19c8a36952b6b710161e6e3f9346ce8476118d9d013b",
            hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
