import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.attachment_identity_compiler import compile_attachment_identities


class AttachmentIdentityCompilerTests(unittest.TestCase):
    def test_groups_exact_socket_identities_without_claiming_resource_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"assets": [{"asset_id": "city/test", "attachments": {"points": [
                {"source_name": "Light_Window", "id": "attachment_00", "bone": 2, "skeleton": 0, "semantic": "night_light", "state_hint": "operational", "bone_local_transform": {"position": [0, 0, 1]}},
                {"source_name": "FX_Fire", "id": "attachment_01", "bone": 3, "skeleton": 0, "semantic": "flame", "state_hint": "operational", "bone_local_transform": {"position": [1, 0, 0]}}
            ]}}]}), encoding="utf-8")
            result = compile_attachment_identities([path])
            self.assertEqual(2, result["summary"]["identities"])
            self.assertTrue(all(item["decoder_status"] == "resource_graph_pending" for item in result["identities"]))
            self.assertEqual("not_enabled", result["runtime_binding"])


if __name__ == "__main__":
    unittest.main()
