import math
import struct
import unittest

from Renderer.tools.asset_compiler.school_orientation import align_school_payload, body_components


class SchoolOrientationTests(unittest.TestCase):
    def fixture(self):
        def bone(name, parent, position):
            return {"name": name, "parent": parent, "local": {"position": position,
                "orientation": [0, 0, 0, 1], "scale_shear": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}
        skeleton = {"bones": [bone("root", -1, [0, 0, 0]), bone("A_head", 0, [-1, 0, 0]),
            bone("A_tail", 1, [2, 0, 0]), bone("B_head", 0, [4, 4, 0]), bone("B_tail", 3, [0, -2, 0])]}
        positions = [(-1, -1, 0), (1, -1, 0), (0, 1, 0), (3, 2, 0), (5, 2, 0), (4, 4, 0)]
        vertices = [{"position": list(p), "joints": [1, 2, 3, 0] if i < 3 else [3, 4, 1, 0],
                     "weights": [.49, .5, .01, 0]} for i, p in enumerate(positions)]
        mesh = {"vertices": vertices, "topology": {"indices": list(range(6))}}
        payload = bytearray(struct.pack("<8s5If", b"C3XANM1\0", 1, 6, 6, 5, 2, 1))
        for v in vertices:
            payload.extend(struct.pack("<8f4I4f", *v["position"], 0, 0, 1, 0, 0, *v["joints"], *v["weights"]))
        payload.extend(struct.pack("<6I", *range(6)))
        for frame in range(2):
            for joint in range(5):
                payload.extend(struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, frame*.1*joint, 0, 0, 1))
        return bytes(payload), mesh, skeleton, [("A_head", "A_tail"), ("B_head", "B_tail")]

    def test_cross_body_weights_keep_the_complete_authored_deformation(self):
        payload, mesh, skeleton, pairs = self.fixture()
        aligned, evidence = align_school_payload(payload, mesh, skeleton, pairs)
        self.assertEqual(len(evidence["bodies"]), 2)
        self.assertEqual(evidence["palette_count"], 6)  # Shared A/B influences have separate transforms.
        offset = 32+6*64+6*4
        for frame in range(2):
            for i, source in enumerate(mesh["vertices"]):
                data = struct.unpack_from("<8f4I4f", aligned, 32+i*64)
                actual = [0, 0, 0]
                for joint, weight in zip(data[8:12], data[12:16]):
                    matrix = struct.unpack_from("<16f", aligned, offset+(frame*6+joint)*64)
                    for a in range(3):
                        actual[a] += weight*(sum(source["position"][b]*matrix[b*4+a] for b in range(3))+matrix[12+a])
                x, y, z = source["position"]
                x += frame*.1*sum(j*w for j, w in zip(source["joints"], source["weights"]))
                # Each center translates with its weighted source influences;
                # calibration changes heading, never the swimming trajectory.
                delta = frame*.1*sum(j*w for j, w in zip(source["joints"], source["weights"]))
                if i < 3:
                    wanted = (2*delta-x, -2/3-y, z)
                else:
                    # B's head/tail receive differing x translations in frame
                    # one, so its heading is no longer exactly vertical.
                    angle = -math.atan2(2, -frame*.1)
                    c, s = math.cos(angle), math.sin(angle)
                    wanted = (4+delta+(x-4-delta)*c-(y-8/3)*s,
                              8/3+(x-4-delta)*s+(y-8/3)*c, z)
                for a in range(3):
                    self.assertAlmostEqual(actual[a], wanted[a], places=5)
        self.assertAlmostEqual(abs(evidence["bodies"][0]["yaw"]), math.pi)
        self.assertAlmostEqual(evidence["bodies"][1]["yaw"], -math.pi/2)

    def test_seam_vertices_are_one_body_and_ambiguous_ownership_is_rejected(self):
        payload, mesh, skeleton, pairs = self.fixture()
        mesh["vertices"].append(dict(mesh["vertices"][0]))
        self.assertEqual([len(c) for c in body_components(mesh)], [4, 3])
        payload, mesh, skeleton, pairs = self.fixture()
        mesh["vertices"][0]["weights"] = [.25, .25, .5, 0]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            align_school_payload(payload, mesh, skeleton, pairs)

    def test_missing_body_and_truncated_payload_fail_before_emission(self):
        payload, mesh, skeleton, pairs = self.fixture()
        with self.assertRaisesRegex(ValueError, "body count"):
            align_school_payload(payload, mesh, skeleton, pairs[:1])
        with self.assertRaisesRegex(ValueError, "length"):
            align_school_payload(payload[:-1], mesh, skeleton, pairs)


if __name__ == "__main__":
    unittest.main()
