"""Members of a baked animal herd never touch, and every animal stays inside its tile.

Herds face outward from their centre, so members placed too close meet at the
hindquarters. Each member's footprint is swept over its whole clip (grazing
moves heads and legs) from the generated animation pack and placed exactly as
the runtime places composition anchors; every pair in every land layout of the
production bake must keep a gap. Skipped when the generated packs are absent.
"""
import json
import math
import struct
import unittest

import numpy as np

from Renderer.lab.platform import ROOT
from Renderer.tools.asset_compiler.build_resource_compositions import TERRAIN_INDEX

ANIMATION = ROOT / "Renderer/packs/ResourceAnimationRuntime"
REPORT = ROOT / "Renderer/packs/ResourceCompositions/composition_report.json"
MOUNTAINS = (1 << TERRAIN_INDEX["Mountains"]) | (1 << TERRAIN_INDEX.get("Volcano", TERRAIN_INDEX["Mountains"]))


def swept(binding, record, cache):
    """Tile-plane points the posed body covers over 24 samples of its clip."""
    if binding not in cache:
        data = (ANIMATION / record["mesh"]).read_bytes()
        _, _, count, index_count, bones, frames, _ = struct.unpack_from("<8s5If", data)
        raw = np.frombuffer(data, np.uint8, count=64 * count, offset=32).reshape(count, 64)
        position = raw[:, 0:12].copy().view(np.float32).reshape(count, 3)
        joints = raw[:, 32:48].copy().view(np.uint32).reshape(count, 4)
        weights = raw[:, 48:64].copy().view(np.float32).reshape(count, 4)
        palettes = np.frombuffer(data, np.float32, count=16 * bones * frames,
                                 offset=32 + 64 * count + 4 * index_count).reshape(frames, bones, 4, 4)
        samples = np.linspace(0, frames - 1, 24).astype(int)
        point = np.concatenate([position, np.ones((count, 1), np.float32)], 1)
        posed = np.zeros((len(samples), count, 3))
        for k in range(4):
            posed += weights[None, :, k, None] * np.einsum("vi,fvij->fvj", point,
                                                           palettes[samples][:, joints[:, k]])[..., :3]
        points = posed.reshape(-1, 3)[:, :2] + [record["offset_x"], record["offset_y"]]
        cache[binding] = points[np.random.RandomState(0).choice(len(points), min(len(points), 1500), replace=False)]
    return cache[binding]


def herd_spacing(report, bindings):
    """(closest gap between members, smallest margin to the tile edge) per herd resource."""
    cache, result = {}, {}
    for name, variants in report["layouts"].items():
        gaps, margins = [], []
        for variant in variants:
            if variant["terrain_mask"] & MOUNTAINS:
                continue   # Conquests never places these animals on mountains
            bodies = []
            for item in variant["instances"]:
                if not item.get("model", "").startswith("animated/"):
                    continue
                binding = item["model"][9:]
                record = bindings[binding]
                points = swept(binding, record, cache)
                yaw, scale = record["yaw"] + item["rotation"], record["scale"] * item["scale"]
                c, s = math.cos(yaw), math.sin(yaw)
                bodies.append(np.stack([item["u"] + (points[:, 0] * c - points[:, 1] * s) * scale,
                                        item["v"] + (points[:, 0] * s + points[:, 1] * c) * scale], 1))
            for a, body in enumerate(bodies):
                margins.append(min(body.min(), 1 - body.max()))
                for other in bodies[a + 1:]:
                    gaps.append(np.sqrt(((body[:, None] - other[None]) ** 2).sum(-1)).min())
        if len(margins):
            result[name] = (min(gaps) if gaps else float("inf"), min(margins))
    return result


@unittest.skipUnless(REPORT.is_file() and (ANIMATION / "bindings.json").is_file(),
                     "requires the prepared resource packs")
class HerdSpacingTests(unittest.TestCase):
    def test_herd_members_keep_apart_and_inside_their_tile(self):
        bindings = json.loads((ANIMATION / "bindings.json").read_text())["bindings"]
        spacing = herd_spacing(json.loads(REPORT.read_text()), bindings)
        self.assertGreaterEqual(set(spacing), {"horses", "cattle", "game", "furs", "ivory"})
        for name, (gap, margin) in spacing.items():
            self.assertGreater(gap, 0.01, f"{name} herd members touch")
            self.assertGreater(margin, 0.0, f"{name} leaves its tile")


if __name__ == "__main__":
    unittest.main()
