"""Incomplete recordings need an explicit, verified comparison boundary."""
import unittest
from Renderer.native.measure_input_replay import inspection_permits_scope


class ReplayScopeTests(unittest.TestCase):
    def test_never_promotes_a_verified_prefix_to_complete(self):
        report = {'complete': False, 'verified_prefix': True, 'last_verified_sequence': 200}
        self.assertFalse(inspection_permits_scope(report, 1, 0))
        self.assertTrue(inspection_permits_scope(report, 1, 150))
        self.assertFalse(inspection_permits_scope(report, 1, 201))
        self.assertFalse(inspection_permits_scope(report, 2, 150))
        self.assertFalse(inspection_permits_scope(report | {'verified_prefix': False}, 1, 150))
        self.assertTrue(inspection_permits_scope(report | {'complete': True}, 0, 0))
        self.assertFalse(inspection_permits_scope(report | {'complete': True}, 1, 0))
