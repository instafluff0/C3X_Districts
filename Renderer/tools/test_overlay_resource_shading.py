import unittest

from Renderer.tools.overlay_resource_shading import resource_hunks


class ResourceOverlayTests(unittest.TestCase):
    def test_only_resource_differences_reach_the_pinned_pack(self):
        # The pinned pack carries another feature's newer overlay (routes); the
        # generated source carries the newer resource clip. Only the latter moves.
        pack = "a\nroute = pinned_overlay;\nb\nclip(mine - 0.08);  // resource bodies\nc\n"
        source = "a\nroute = older_source;\nb\nclip(mine - decal);  // resource bodies\nfloat resource_decal_weight;\nc\n"
        result, applied = resource_hunks(pack, source)
        self.assertEqual(applied, 1)
        self.assertEqual(result, "a\nroute = pinned_overlay;\nb\nclip(mine - decal);  // resource bodies\n"
                                 "float resource_decal_weight;\nc\n")
        self.assertEqual(resource_hunks(result, source), (result, 0))   # idempotent once current

    def test_a_merged_neighbouring_change_is_shown_for_review(self):
        # Adjacent changed lines form one hunk; it is taken whole and listed.
        shown = []
        result, applied = resource_hunks("route = pinned;\nclip(old); // resource\n",
                                         "route = source;\nclip(new); // resource\n", shown)
        self.assertEqual(applied, 1)
        self.assertIn("- route = pinned;", shown[0])

    def test_match_excludes_other_work_that_mentions_resources(self):
        # A peer's farm change edits a comment about resource bodies; only the
        # marked resource shadow hunk may reach the game.
        pack = "// resource bodies\nx\nshadow = old;  // resource shadow\n"
        source = "// resource bodies and farm kits\nx\nshadow = new;  // resource shadow lies on the ground\n"
        result, applied = resource_hunks(pack, source, match="lies on the ground")
        self.assertEqual((result, applied), ("// resource bodies\nx\nshadow = new;  // resource shadow lies on the ground\n", 1))


if __name__ == "__main__":
    unittest.main()
