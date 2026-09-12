"""Keep the short causal batch from turning variation or pixel omissions into a win."""
import unittest
from Renderer.native.diagnose_dense_navigation import compare_runs, summarize


def run(value, image="exact"):
    return {"mean_transition_ms": value, "phase_mean_ms": {"geometry": value / 2},
            "images": {"initial": image, "final": image}}


class CausalDecisionTests(unittest.TestCase):
    def test_large_matched_reduction_and_small_or_overlapping_effects(self):
        baseline = [run(300), run(330)]
        self.assertEqual(compare_runs(baseline, [run(200), run(215)])["decision"], "useful_causal_effect")
        self.assertEqual(compare_runs(baseline, [run(295), run(325)])["decision"], "reject_as_primary_target")
        self.assertEqual(compare_runs(baseline, [run(265), run(310)])["decision"], "inconclusive_repeat_variation")
        self.assertEqual(compare_runs([run(300)], [run(200)])["decision"], "inconclusive_insufficient_repetitions")

    def test_prepared_content_requires_same_fresh_pixels(self):
        runs = []
        for repeat in range(2):
            for arm, image in (("full", "exact"), ("prepared_content", "changed")):
                runs.append(dict(run(300, image), repeat=repeat, workload="four_columns", arm=arm))
        with self.assertRaisesRegex(ValueError, "Prepared content differs"):
            summarize(runs)

    def test_omitted_pixels_cannot_be_correctness_pass(self):
        runs = []
        for repeat in range(2):
            for arm, value, image in (("full", 300, "exact"), ("route_draws_omitted", 200, "omitted")):
                runs.append(dict(run(value, image), repeat=repeat, workload="four_columns", arm=arm))
        result = summarize(runs)["four_columns/route_draws_omitted"]
        self.assertEqual(result["decision"], "useful_causal_effect")
        self.assertIn("cannot pass production correctness", result["correctness"])


if __name__ == "__main__":
    unittest.main()
