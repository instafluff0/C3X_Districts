from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.state_provenance_compiler import DEFAULT_PROVENANCE, compile_state_provenance


class StateProvenanceCompilerTests(unittest.TestCase):
    def test_every_scene_field_has_authoritative_or_bounded_audit_provenance(self) -> None:
        result = compile_state_provenance()
        self.assertEqual(result["summary"]["scene_fields"], result["summary"]["covered_fields"])
        self.assertGreater(result["summary"]["gate_audits"], 0)
        self.assertEqual(0, result["summary"]["new_patch_requests"])
        self.assertEqual([], result["required_user_action"])

    def test_unproved_patch_request_is_rejected(self) -> None:
        value = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
        value["required_user_action"] = [{"symbol": "speculative"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unproved patch request"):
                compile_state_provenance(provenance_path=path)


if __name__ == "__main__":
    unittest.main()
