"""Actual changed bytes select review scope, including contextual consumers."""
import unittest
from unittest.mock import patch

from Renderer import renderer
from Renderer.lab import dependencies


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.entries = {key: {"recipe": {"objects": key in ("cities", "day-night", "shadows", "animation")},
                             "depends_on": []}
                        for key in ("grassland", "cities", "units", "resources", "animation", "day-night", "shadows")}

    def changed(self, path, *, assets=None, generated=()):
        before = dependencies.signatures({path: "before"}, self.entries, assets=assets, generated_shaders=generated)
        after = dependencies.signatures({path: "after"}, self.entries, assets=assets, generated_shaders=generated)
        return {key for key in before if before[key] != after[key]}

    def test_local_city_light_selects_actual_city_contexts_not_every_terrain(self):
        self.assertEqual(self.changed("Renderer/lab/shared/shaders/lighting/local_facade_lights.hlsl"),
                         {"cities", "day-night", "shadows", "animation"})

    def test_shared_lighting_shadow_transition_and_unknown_code_select_every_consumer(self):
        for path in ("Renderer/native/environment_runtime.cpp",
                     "Renderer/lab/shared/shaders/lighting/shadow_visibility_v1.hlsl",
                     "Renderer/lab/shared/shaders/terrain/source_blend.hlsl",
                     "Renderer/lab/shared/new_system.py"):
            self.assertEqual(self.changed(path), set(self.entries))

    def test_unit_edit_selects_units_animation_and_mixed_shadows(self):
        self.assertEqual(self.changed("Renderer/packs/UnitAnimationRuntime/clips/test.bin"), {"units", "animation", "shadows"})

    def test_resource_edit_includes_resource_bearing_contexts(self):
        self.assertEqual(self.changed("Renderer/lab/shared/resources/clip_units.json"),
                         {"resources", "cities", "day-night", "shadows", "animation"})

    def test_observed_input_shared_by_builders_unions_consumers(self):
        path = "Renderer/tools/asset_compiler/normalized_animation.py"
        assets = {name: {"inputs": {path: "old"}} for name in ("units", "resources")}
        self.assertEqual(self.changed(path, assets=assets), set(self.entries) - {"grassland"})

    def test_preparation_changes_remain_global_even_with_one_cached_job(self):
        path = "Renderer/lab/asset_preparation.py"
        self.assertEqual(self.changed(path, assets={"units": {"inputs": {path: "old"}}}), set(self.entries))

    def test_generated_shader_text_does_not_broaden_source_scope(self):
        path = "Renderer/native/city_fidelity/terrain.hlsl"
        self.assertEqual(self.changed(path, generated=[path]), set())
        # Without an established preparation contract, take the safe broad path.
        self.assertEqual(self.changed(path), set(self.entries))

    def test_added_and_removed_inputs_change_signatures(self):
        before = dependencies.signatures({}, self.entries)
        after = dependencies.signatures({"Renderer/packs/UnitAnimationRuntime/new.bin": "new"}, self.entries)
        self.assertEqual({k for k in before if before[k] != after[k]}, {"units", "animation", "shadows"})

    def test_recipe_is_part_of_selection(self):
        current = dependencies.signatures({}, self.entries)
        updated = {key: {**value, "recipe": dict(value["recipe"])} for key, value in self.entries.items()}
        updated["grassland"]["recipe"]["terrain"] = 1
        self.assertNotEqual(current["grassland"], dependencies.signatures({}, updated)["grassland"])

    def test_requested_category_and_integration_use_declared_dependents_only(self):
        with patch.object(renderer, "declared_affected", return_value=["grassland"]):
            self.assertEqual(renderer.affected("grassland"), ["grassland"])

    def test_asset_preparation_is_category_scoped(self):
        with patch.object(renderer, "catalog", return_value=self.entries), \
             patch.object(renderer, "standard", side_effect=self.entries.__getitem__):
            self.assertEqual(renderer.asset_jobs_for(["grassland"]), {"natural", "hill-cliff", "coastal-waves"})
            self.assertEqual(renderer.asset_jobs_for(["units"]), {"natural", "hill-cliff", "coastal-waves", "units"})

    def test_asset_preparation_is_category_scoped(self):
        with patch.object(renderer, "catalog", return_value=self.entries), \
             patch.object(renderer, "standard", side_effect=self.entries.__getitem__):
            self.assertEqual(renderer.asset_jobs_for(["grassland"]), {"natural", "hill-cliff", "coastal-waves"})
            self.assertEqual(renderer.asset_jobs_for(["units"]), {"natural", "hill-cliff", "coastal-waves", "units"})


if __name__ == "__main__":
    unittest.main()
