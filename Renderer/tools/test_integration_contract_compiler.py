import unittest

from Renderer.tools.integration_contract_compiler import cache_key, compile_contracts


class IntegrationContractCompilerTests(unittest.TestCase):
    def test_all_categories_are_inactive_and_fail_without_native_replay(self) -> None:
        result = compile_contracts()
        self.assertEqual("none", result["runtime_activation"])
        self.assertEqual(0, result["injected_code_changes"])
        self.assertEqual(
            {"resources", "cities", "mines", "farms", "tile_objects", "infrastructure", "units"},
            set(result["categories"]),
        )
        for contract in result["categories"].values():
            self.assertEqual("not_enabled", contract["activation"])
            self.assertEqual(
                "custom_category_hard_failure_no_native_replay",
                contract["ownership"]["enabled_failure"],
            )
            self.assertTrue(contract["invalidation_fixture_keys"])

    def test_cache_key_is_order_independent_and_complete(self) -> None:
        result = compile_contracts()
        resource = result["categories"]["resources"]
        recipe = {"cache_inputs": [field for _, field in resource["cache_key_recipe"]]}
        forward = dict(resource["fixture"])
        reverse = dict(reversed(list(forward.items())))
        self.assertEqual(cache_key(recipe, forward), cache_key(recipe, reverse))


if __name__ == "__main__":
    unittest.main()
