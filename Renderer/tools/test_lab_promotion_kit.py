import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.lab_promotion_kit import build_promotion_kit, write_promotion_kit


class LabPromotionKitTests(unittest.TestCase):
    def test_every_future_profile_has_two_zoom_and_full_selector_state_coverage(self) -> None:
        for gate in ("L16", "L17", "L18", "L19", "L19A", "L19B", "L20"):
            kit = build_promotion_kit(gate)
            self.assertEqual(192, len(kit["fixture"]["tiles"]))
            self.assertEqual({"normal", "reduced"}, {case["zoom"] for case in kit["cases"]})
            self.assertEqual("draft_not_approved", kit["handoff_status"])
            self.assertEqual("not_transferred", kit["integration_authority"])

    def test_output_is_deterministic_and_cannot_fabricate_approval(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            write_promotion_kit("L16", Path(first))
            write_promotion_kit("L16", Path(second))
            self.assertEqual(
                (Path(first) / "promotion_matrix.json").read_bytes(),
                (Path(second) / "promotion_matrix.json").read_bytes(),
            )
            handoff = json.loads((Path(first) / "handoff_draft.json").read_text())
            self.assertIsNone(handoff["approval"])
            self.assertEqual({}, handoff["artifact_hashes"])


if __name__ == "__main__":
    unittest.main()
