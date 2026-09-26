import unittest
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops
from Renderer.lab.studies.borders.mesh_surface import GroundSurface, RenderDepth

from Renderer.lab.studies.borders.preview import (
    CASES, CITY, NEIGHBORS, arc_lengths, biq_city_territory,
    city_territory, coherent_occlusion, compose, diamond, draped_paths,
    exposed_edges, inward_fade_mask, overlay,
    perimeter_loops, point_at, read_biq_terrain, round_tile_corners,
    terrain_visibility,
)


class BorderStudyTests(unittest.TestCase):
    def test_only_outer_edges_of_parity_valid_city_territory(self):
        owned = city_territory()
        self.assertIn(CITY, owned)
        self.assertTrue(all((x + y) % 2 == 0 for x, y in owned))
        edges = set(exposed_edges(owned))
        self.assertTrue(edges)
        for x, y, side in edges:
            dx, dy = NEIGHBORS[side]
            self.assertNotIn((x + dx, y + dy), owned)
        self.assertFalse(any((x + dx, y + dy) in owned and (x, y, side) in edges
                             for x, y in owned for side, (dx, dy) in enumerate(NEIGHBORS)))
        loops = perimeter_loops(owned, 128, (640, 480))
        self.assertEqual(1, len(loops))
        self.assertEqual(len(edges), len(loops[0]))

    def test_continuous_single_color_strokes_at_both_zooms(self):
        for zoom in (64, 128):
            color, width = CASES["crimson-brush"]
            layer = overlay((640, 480), zoom, color, width)
            alpha = layer.getchannel("A")
            self.assertIsNotNone(alpha.getbbox())
            center = diamond(*CITY, zoom, (640, 480))
            self.assertEqual(0, alpha.getpixel((320, 240)))
            curve = round_tile_corners(perimeter_loops(city_territory(), zoom, (640, 480))[0])
            distances = arc_lengths(curve)
            samples = [point_at(curve, distances, distances[-1] * i / 500) for i in range(500)]
            coverage = [alpha.getpixel((round(x), round(y))) > 70 for x, y in samples]
            self.assertTrue(all(coverage))
            self.assertEqual(center[0][0], 320)
            colored = [pixel for pixel in layer.get_flattened_data() if pixel[3] > 0]
            self.assertTrue(all(pixel[:3] == color for pixel in colored))

    def test_opacity_fades_outward_from_the_same_color_center(self):
        color, width = CASES["crimson-brush"]
        layer = overlay((640, 480), 128, color, width)
        alpha = layer.getchannel("A")
        path = draped_paths(city_territory(), (640, 480), 128, CITY)[0]
        index = len(path) // 4
        x, y = path[index]
        before, after = path[index - 4], path[index + 4]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = (dx*dx + dy*dy) ** 0.5
        nx, ny = -dy/length, dx/length
        sample = lambda offset: alpha.getpixel((round(x + nx*offset), round(y + ny*offset)))
        self.assertGreater(sample(0), sample(width*0.8))
        self.assertGreater(sample(width*0.8), sample(width*2.5))

    def test_inward_band_fades_with_varied_opacity_only_on_owned_ground(self):
        size = (640, 480)
        flat_paths = []
        draped_paths({CITY}, size, 128, CITY, flat_paths=flat_paths)
        wash = inward_fade_mask(flat_paths, size, 4.2, 3)
        def sample(distance, inside):
            sign = -1 if inside else 1
            return wash.getpixel((round(352+sign*.447*distance),
                                  round(224-sign*.894*distance)))
        self.assertGreater(sample(5, True), 35)
        self.assertLess(sample(5, True), 110)
        self.assertGreater(sample(5, True), sample(10, True))
        self.assertGreater(sample(10, True), sample(14, True))
        self.assertLess(sample(17, True), sample(5, True)//4)
        self.assertEqual(0, sample(8, False))
        along_edge = [wash.getpixel((round(x-4*.447), round(y+4*.894)))
                      for x, y in ((336, 216), (348, 222), (360, 228), (372, 234))]
        self.assertGreaterEqual(max(along_edge)-min(along_edge), 4)

    def test_inward_band_uses_ground_projection_and_local_occlusion(self):
        size = (640, 480)
        flat_paths = []
        draped_paths({CITY}, size, 128, CITY, flat_paths=flat_paths)
        class RaisedGround:
            def project_with_depth(self, x, y, image_size, tile_width, center):
                return x, y-12, 0
        class Foreground:
            def is_occluded(self, x, y, depth, clearance=5.0):
                return 345 < x < 360
        base = inward_fade_mask(flat_paths, size, 4.2, 3)
        draped = inward_fade_mask(flat_paths, size, 4.2, 3,
                                  surface=RaisedGround())
        hidden = inward_fade_mask(flat_paths, size, 4.2, 3,
                                  surface=RaisedGround(), occluders=Foreground())
        self.assertGreater(base.getpixel((350, 222)), 90)
        self.assertLessEqual(base.getpixel((350, 210)), 2)
        self.assertAlmostEqual(base.getpixel((350, 222)),
                               draped.getpixel((350, 210)), delta=20)
        self.assertLess(hidden.getpixel((350, 210)), draped.getpixel((350, 210))//2)

    def test_occlusion_ignores_single_spots_but_keeps_solid_crossings(self):
        self.assertEqual([False]*7,
                         coherent_occlusion([False, False, True, False, False, False, False]))
        self.assertEqual([False, True, True, True, True, True, False],
                         coherent_occlusion([False, True, True, False, True, True, False]))
        candidates = [False, True, True, True, True, True, True, False]
        self.assertEqual([False]*8,
                         coherent_occlusion(candidates,
                                            [False, False, True, False, True, False, False, False]))
        self.assertEqual(candidates,
                         coherent_occlusion(candidates,
                                            [False, True, True, True, True, False, False, False]))

    def test_color_changes_only_the_border(self):
        source = Image.new("RGB", (640, 480), (80, 105, 65))
        red = compose(source, 128, "crimson-brush")
        blue = compose(source, 128, "azure-brush")
        self.assertIsNotNone(ImageChops.difference(red, blue).getbbox())
        self.assertEqual(source.getpixel((320, 240)), red.getpixel((320, 240)))
        self.assertEqual(red.tobytes(), compose(source, 128, "crimson-brush").tobytes())
        green = compose(source, 128, "crimson-brush", (40, 180, 70))
        self.assertIsNotNone(ImageChops.difference(red, green).getbbox())

    def test_biq_region_uses_real_map_tiles_and_keeps_straight_edge_centers(self):
        site = (20, 64)
        selected = {(site[0] + dc + dr, site[1] + dc - dr)
                    for dc in range(-1, 2) for dr in range(-1, 2)} | {(22, 66)}
        with TemporaryDirectory() as directory:
            csv = Path(directory) / "terrain.csv"
            csv.write_text("C3X_BIQ_TERRAIN_V3,100,100,11\n" +
                           "\n".join(f"{x},{y},2,2,0,0,0" for x, y in sorted(selected)) + "\n")
            owned = biq_city_territory(site, csv)
            terrain = read_biq_terrain(csv)
        self.assertEqual(selected, owned)
        self.assertEqual((2, 2), terrain[site])
        corners = perimeter_loops(owned, 256, (1280, 800), site)[0]
        rounded = round_tile_corners(corners)
        self.assertEqual(9 * len(corners), len(rounded))
        # The interval from departure to the next approach remains exactly on
        # the exposed straight tile edge, unlike a global smoothing spline.
        for index in range(len(corners)):
            departure = rounded[index * 9 + 8]
            approach = rounded[((index + 1) % len(corners)) * 9]
            a, b = corners[index], corners[(index + 1) % len(corners)]
            cross = (b[0] - a[0]) * (approach[1] - departure[1]) - (b[1] - a[1]) * (approach[0] - departure[0])
            self.assertAlmostEqual(0, cross, places=5)

    def test_projected_mesh_samples_the_mountain_replacement_layer(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "surface"
            payload = bytearray(b"C3XBRD1\0" + struct.pack("<2i", 16, 16))
            for lift in (0, 12):
                vertices = []
                for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    vertices.extend((16+u, 1-v, (2.5+lift)/112))
                payload.extend(struct.pack("<2I", 4, 6))
                payload.extend(struct.pack("<12f", *vertices))
                payload.extend(struct.pack("<6I", 0, 1, 2, 0, 2, 3))
            (Path(directory) / "surface.16_16.bin").write_bytes(payload)
            surface = GroundSurface(prefix, 128)
            self.assertAlmostEqual(14.5/112, surface.sample_world(16.5, 0.5))
            x, y = surface.project(320, 240, (640, 480), 128, CITY)
            self.assertAlmostEqual(320, x)
            self.assertAlmostEqual(240-12*(128/224*0.82), y, places=6)
            with self.assertRaisesRegex(ValueError, "misses exported ground mesh"):
                surface.sample_world(19.5, 0.5)

    def test_renderer_depth_dulls_only_a_foreground_crossing(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "surface"
            rows = [0.49 if 40 <= x < 60 else
                    0.5-1.5/16384 if x == 20 and y == 50 else 0.5
                    for y in range(100) for x in range(100)]
            (Path(directory) / "surface.depth.0_0.bin").write_bytes(
                b"C3XBDP1\0" + struct.pack("<4if", 0, 0, 100, 100, 0.0) +
                struct.pack("<10000f", *rows))
            depth = RenderDepth(prefix, (100, 100))
            self.assertTrue(depth.is_occluded(50, 50, 0.0))
            self.assertFalse(depth.is_occluded(20, 50, 0.0))
            path = [(float(x), 50.0) for x in range(10, 91, 2)]
            visibility = terrain_visibility([path], [[0.0]*len(path)], depth,
                                            (100, 100), 4.0)
            self.assertEqual(255, visibility.getpixel((20, 50)))
            self.assertLess(visibility.getpixel((50, 50)), 100)
            self.assertEqual(255, visibility.getpixel((80, 50)))


if __name__ == "__main__":
    unittest.main()
