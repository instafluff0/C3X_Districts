"""The site depth overlay brings the checkout's site lines to a pack's VSSharedFeature."""
import tempfile
import unittest
from pathlib import Path

from Renderer.tools import overlay_site_depth as tool

# A pack copy from before the tunnel and site lines (the pinned pack's form).
PACK_SHADER = """FeaturePixelInput VSSharedFeature(RigidInput i){
 bool bridge=i.projection.z!=0 && i.placement_view.w>12.5 && i.placement_view.w<20.5;
 // Farm kit trees and farmhouses stand on low relief and hills in the same way.
 bool farm=i.projection.z!=0 && farm_kit_material(i.placement_view.w);
 if(bridge || farm){float h=p.world.z*112-2.5;
  position.z=position.y+h*(i.projection.z/224*.82)+h*.0016*i.projection.w;}
 return o;
}
"""


class SiteDepthOverlayTests(unittest.TestCase):
    def test_site_lines_join_the_pack_basis_once(self):
        lines = tool.site_lines((tool.ROOT / tool.SOURCE).read_text())
        self.assertIn("bool site=i.projection.z!=0 && i.placement_view.w>20.5 && i.placement_view.w<28.5", lines)
        with tempfile.TemporaryDirectory() as folder:
            pack = Path(folder) / "pack"
            shader = pack / "Renderer/native/city_fidelity/rigid_feature.hlsl"
            shader.parent.mkdir(parents=True)
            shader.write_text(PACK_SHADER)
            (pack / "other.hlsl").write_text("float4 PSOther():SV_Target{return 0;}\n")
            self.assertEqual(["Renderer/native/city_fidelity/rigid_feature.hlsl"],
                             sorted(tool.overlay(pack, dry_run=True)))
            record = tool.overlay(pack, Path(folder) / "before")
            text = shader.read_text()
            self.assertIn(lines + " if(bridge || farm || site){float h=p.world.z*112-2.5;\n", text)
            self.assertEqual(PACK_SHADER, (Path(folder) / "before" / record["outputs"].popitem()[0]).read_text())
            self.assertEqual(text.replace(lines, "").replace(" || site", ""), PACK_SHADER)
            # Applied once: a second run finds the pack current.
            self.assertIsNone(tool.overlay(pack, Path(folder) / "again"))


if __name__ == "__main__":
    unittest.main()
