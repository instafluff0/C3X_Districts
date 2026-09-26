import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops

from Renderer.lab.studies.borders.preview import (
    CASES, CITY, NEIGHBORS, ReliefStudy, arc_lengths, biq_city_territory,
    city_territory, compose, diamond, draped_paths, exposed_edges, overlay,
    perimeter_loops, point_at, read_biq_terrain, round_tile_corners,
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
                    for dc in range(-1, 2) for dr in range(-1, 2)} | {(22, 66), (21, 67)}
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

    def test_biq_relief_drapes_the_tile_boundary(self):
        owned = city_territory()
        terrain = {tile: (2, 5) for tile in owned}
        relief = ReliefStudy(terrain, 128, (640, 480), CITY)
        flat = draped_paths(owned, (640, 480), 128, CITY)
        raised = draped_paths(owned, (640, 480), 128, CITY, relief)
        self.assertEqual([len(path) for path in flat], [len(path) for path in raised])
        self.assertTrue(any(y-flat_y < -8 for path, original in zip(raised, flat)
                            for (_, y), (_, flat_y) in zip(path, original)))
        self.assertEqual([x for path in flat for x, _ in path],
                         [x for path in raised for x, _ in path])

    def test_forest_does_not_break_the_border(self):
        owned = city_territory()
        x, y, _ = next(iter(exposed_edges(owned)))
        forest = ReliefStudy({(x, y): (2, 7)}, 128, (640, 480), CITY)
        flat = draped_paths(owned, (640, 480), 128, CITY)
        covered = draped_paths(owned, (640, 480), 128, CITY, forest)
        self.assertEqual(flat, covered)
        self.assertEqual(1, len(covered))
        self.assertEqual(covered[0][0], covered[0][-1])


if __name__ == "__main__":
    unittest.main()
