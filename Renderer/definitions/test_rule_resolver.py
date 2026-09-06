import unittest
from pathlib import Path

from Renderer.definitions.definition_parser import merge_layers, parse_definitions
from Renderer.definitions.rule_resolver import coordinate_variant_seed, resolve_rule


BASE = """
#Profile
id = default
terrain = replace
features = replace
roads = civ3
rivers = civ3
improvements = civ3
resources = replace
cities = replace
units = replace
effects = civ3
missing_asset = fallback

#Pack
id = base
path = mod:packs/base

#Asset
id = shared
pack = base
asset = materials/shared.json

#Asset
id = grass
pack = base
asset = materials/grass.json

#Asset
id = horse
pack = base
asset = materials/horse.json

#Asset
id = roman_city
pack = base
asset = meshes/roman_city.glb

#Asset
id = warrior
pack = base
asset = meshes/warrior.glb
"""


def catalog_with(rules: str):
    definitions = parse_definitions(BASE + rules, Path("default.custom_rendering.txt"), "default", Path.cwd(), Path.cwd())
    return merge_layers([("default", definitions)])


class RuleResolverTests(unittest.TestCase):
    def test_documented_shared_and_terrain_selector_vocabulary(self):
        catalog = catalog_with(
            """
#Rule
id = exhaustive_terrain
category = terrain
map_x = 4
map_y = 8
landmark = true
owner = Rome
civilization = Romans
era = Ancient Times
show_in_day_night_hours = 18-5
show_in_seasons = autumn,winter
adjacent_to = coast
terrain_type = grassland
real_terrain_type = plains
sheet_index = 2
sprite_index = 7
pcx_file = xgc.pcx
pcx_index = 23
river_mask = 1
road_mask = 2
railroad_mask = 4
has_forest = true
has_jungle = false
has_marsh = false
has_pollution = true
has_crater = false
improvement = mine
terrain_building = airfield
coast_shape = convex
neighbor_mask = 31
asset = grass
"""
        )
        metadata = {
            "category": "terrain",
            "map_x": 4,
            "map_y": 8,
            "landmark": True,
            "owner": "rome",
            "civilization": "romans",
            "era": "ancient times",
            "hour": 23,
            "season": "fall",
            "adjacent_to": ["coast", "river"],
            "terrain_type": "grassland",
            "real_terrain_type": "plains",
            "sheet_index": 2,
            "sprite_index": 7,
            "pcx_file": "XGC.PCX",
            "pcx_index": 23,
            "river_mask": 1,
            "road_mask": 2,
            "railroad_mask": 4,
            "has_forest": True,
            "has_jungle": False,
            "has_marsh": False,
            "has_pollution": True,
            "has_crater": False,
            "improvement": "mine",
            "terrain_building": "airfield",
            "coast_shape": "convex",
            "neighbor_mask": 31,
        }
        result = resolve_rule(catalog, metadata)
        self.assertEqual("exhaustive_terrain", result["winner"]["rule_id"])
        self.assertEqual(27, result["winner"]["rank"]["specificity"])
        self.assertEqual(27, len(next(c for c in result["candidates"] if c["status"] == "winner")["matched_selectors"]))

    def test_representative_selector_families_choose_one_winner(self):
        catalog = catalog_with(
            """
#Rule
id = terrain_general
category = terrain
terrain_type = grassland
asset = shared

#Rule
id = terrain_sheet
category = terrain
terrain_type = grassland
pcx_file = xgc.pcx
sheet_index = 3
sprite_index = 7
asset = grass

#Rule
id = resource_horse
category = resource
resource_id = 5
resource_name = Horses
resource_class = strategic
asset = horse

#Rule
id = city_roman
category = city
culture_group = mediterranean
era = ancient
city_size = town
has_walls = true
is_capital = false
asset = roman_city

#Rule
id = unit_warrior
category = unit
unit_id = 2
unit_type = Warrior
unit_class = land
direction = southeast
action = fortify
fortified = true
hit_point_band = healthy
owner = Rome
civilization = Romans
asset = warrior
animation = warrior_fortify
"""
        )
        cases = [
            (
                {"category": "terrain", "terrain_type": "GRASSLAND", "pcx_file": "XGC.PCX", "sheet_index": 3, "sprite_index": 7},
                "terrain_sheet",
            ),
            (
                {"category": "resource", "resource_id": 5, "resource_name": "horses", "resource_class": "STRATEGIC"},
                "resource_horse",
            ),
            (
                {"category": "city", "culture_group": "Mediterranean", "era": "Ancient", "city_size": "Town", "has_walls": True, "is_capital": False},
                "city_roman",
            ),
            (
                {"category": "unit", "unit_id": 2, "unit_type": "warrior", "unit_class": "land", "direction": "southeast", "action": "fortify", "fortified": True, "hit_point_band": "healthy", "owner": "rome", "civilization": "romans"},
                "unit_warrior",
            ),
        ]
        for metadata, expected in cases:
            with self.subTest(expected=expected):
                result = resolve_rule(catalog, metadata)
                self.assertEqual("matched", result["status"])
                self.assertEqual(expected, result["winner"]["rule_id"])
                self.assertEqual(1, sum(c["status"] == "winner" for c in result["candidates"]))

    def test_ranking_order_and_loser_reasons_are_exact(self):
        default = parse_definitions(
            BASE
            + """
#Rule
id = low_priority
category = terrain
terrain_type = grassland
priority = 1
asset = shared

#Rule
id = low_specificity
category = terrain
terrain_type = grassland
priority = 2
asset = shared

#Rule
id = old_layer
category = terrain
terrain_type = grassland
real_terrain_type = grassland
priority = 2
asset = shared
""",
            Path("default.custom_rendering.txt"),
            "default",
            Path.cwd(),
            Path.cwd(),
        )
        custom = parse_definitions(
            """
#Rule
id = early_declaration
category = terrain
terrain_type = grassland
real_terrain_type = grassland
priority = 2
asset = shared

#Rule
id = winner
category = terrain
terrain_type = grassland
real_terrain_type = grassland
priority = 2
asset = grass
""",
            Path("custom.custom_rendering.txt"),
            "custom",
            Path.cwd(),
            Path.cwd(),
        )
        catalog = merge_layers([("default", default), ("custom", custom)])
        result = resolve_rule(catalog, {"category": "terrain", "terrain_type": "grassland", "real_terrain_type": "grassland"})
        self.assertEqual("winner", result["winner"]["rule_id"])
        reasons = {candidate["rule_id"]: candidate.get("loser_reason") for candidate in result["candidates"]}
        self.assertEqual("lower_priority", reasons["low_priority"])
        self.assertEqual("lower_specificity", reasons["low_specificity"])
        self.assertEqual("lower_layer_precedence", reasons["old_layer"])
        self.assertEqual("earlier_declaration", reasons["early_declaration"])

    def test_wrapped_hours_season_alias_and_adjacency(self):
        catalog = catalog_with(
            """
#Rule
id = night_forest
category = terrain
terrain_type = grassland
adjacent_to = coast
show_in_day_night_hours = 18-5
show_in_seasons = autumn,winter
asset = grass
"""
        )
        result = resolve_rule(
            catalog,
            {"category": "terrain", "terrain_type": "grassland", "adjacent_to": ["river", "COAST"], "hour": 2, "season": "fall"},
        )
        self.assertEqual("matched", result["status"])
        rejected = resolve_rule(
            catalog,
            {"category": "terrain", "terrain_type": "grassland", "adjacent_to": ["coast"], "hour": 12, "season": "spring"},
        )
        self.assertEqual("no_matching_rule", rejected["fallback"]["reason"])
        failures = rejected["candidates"][0]["failed_selectors"]
        self.assertEqual({"show_in_day_night_hours", "show_in_seasons"}, {failure["key"] for failure in failures})

    def test_coordinate_hash_is_stable_and_uses_coordinates_seed_and_rule(self):
        first = coordinate_variant_seed("grass", 10, 20, 99)
        self.assertEqual(first, coordinate_variant_seed("grass", 10, 20, 99))
        self.assertNotEqual(first, coordinate_variant_seed("grass", 11, 20, 99))
        self.assertNotEqual(first, coordinate_variant_seed("grass", 10, 20, 100))
        self.assertNotEqual(first, coordinate_variant_seed("other", 10, 20, 99))

        catalog = catalog_with(
            """
#Rule
id = varied_grass
category = terrain
terrain_type = grassland
variant_selection = coordinate-hash
asset = grass
"""
        )
        result = resolve_rule(catalog, {"category": "terrain", "terrain_type": "grassland", "map_x": 10, "map_y": 20}, world_seed=99)
        self.assertEqual(coordinate_variant_seed("varied_grass", 10, 20, 99), result["winner"]["variant"]["seed"])
        self.assertEqual(0, result["asset_payload_loads"])

    def test_missing_coordinates_fall_back_for_coordinate_variant(self):
        catalog = catalog_with(
            """
#Rule
id = varied_grass
category = terrain
terrain_type = grassland
variant_selection = coordinate-hash
asset = grass
"""
        )
        result = resolve_rule(catalog, {"category": "terrain", "terrain_type": "grassland"})
        self.assertEqual("missing_variant_coordinates", result["fallback"]["reason"])

    def test_missing_asset_and_no_match_fall_back_without_payload_loads(self):
        catalog = catalog_with(
            """
#Rule
id = grass
category = terrain
terrain_type = grassland
asset = grass
"""
        )
        missing = resolve_rule(catalog, {"category": "terrain", "terrain_type": "grassland"}, available_assets={"shared"})
        self.assertEqual("missing_asset", missing["fallback"]["reason"])
        self.assertEqual("grass", missing["winner"]["asset_id"])
        self.assertEqual(1, missing["asset_availability_checks"])
        self.assertEqual(0, missing["asset_payload_loads"])

        unmatched = resolve_rule(catalog, {"category": "terrain", "terrain_type": "desert"})
        self.assertEqual("no_matching_rule", unmatched["fallback"]["reason"])
        self.assertEqual(0, unmatched["asset_payload_loads"])

    def test_config_off_and_civ3_owned_categories_do_no_work(self):
        catalog = catalog_with(
            """
#Rule
id = grass
category = terrain
terrain_type = grassland
asset = grass

#Rule
id = road
category = road
road_mask = 3
asset = shared
"""
        )
        disabled = resolve_rule(catalog, {"category": "terrain", "terrain_type": "grassland"}, enabled=False, available_assets=set())
        self.assertEqual("config_off", disabled["fallback"]["reason"])
        self.assertEqual([], disabled["candidates"])
        self.assertEqual(0, disabled["asset_availability_checks"])
        self.assertEqual(0, disabled["asset_payload_loads"])

        road = resolve_rule(catalog, {"category": "road", "road_mask": 3})
        self.assertEqual("category_owned_by_civ3", road["fallback"]["reason"])
        self.assertEqual(0, road["asset_availability_checks"])

    def test_coordinate_selectors_and_variant_coordinates_are_independent(self):
        catalog = catalog_with(
            """
#Rule
id = landmark
category = terrain
map_x = 4
map_y = 8
landmark = true
asset = grass
"""
        )
        result = resolve_rule(catalog, {"category": "terrain", "map_x": 4, "map_y": 8, "landmark": True})
        self.assertEqual("landmark", result["winner"]["rule_id"])
        self.assertEqual(3, result["winner"]["rank"]["specificity"])

    def test_explicit_disabled_false_is_control_metadata_not_a_selector(self):
        catalog = catalog_with(
            """
#Rule
id = grass
disabled = false
category = terrain
terrain_type = grassland
asset = grass
"""
        )
        result = resolve_rule(catalog, {"category": "terrain", "terrain_type": "grassland"})
        self.assertEqual("matched", result["status"])
        self.assertEqual(1, result["winner"]["rank"]["specificity"])


if __name__ == "__main__":
    unittest.main()
