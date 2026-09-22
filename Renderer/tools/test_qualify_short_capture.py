"""Build provenance remains mandatory across the supported receipt formats."""
import unittest
from Renderer.tools.qualify_short_capture import validate_build


class BuildReceiptTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {'renderer': {'source.cpp': 'source-hash'}}
        self.base = {'returncode': 0, 'sources_unchanged': True, 'unit_inputs': self.inputs}

    def test_standard_and_contract_builder_receipts(self):
        validate_build(self.base | {'binaries': {'C3XRenderer.dll': 'dll'},
                                   'unit_inputs': self.inputs | {'preview': {'preview.cpp': 'other'}}},
                       'dll', self.inputs)
        validate_build(self.base | {'dll_sha256': 'dll'}, 'dll', self.inputs)

    def test_missing_changed_or_conflicting_binary_identity(self):
        for fields in ({}, {'dll_sha256': 'other'},
                       {'binaries': {'C3XRenderer.dll': 'other'}},
                       {'dll_sha256': 'dll', 'binaries': {'C3XRenderer.dll': 'other'}}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_build(self.base | fields, 'dll', self.inputs)

    def test_failed_stale_and_preview_only_builds_rejected(self):
        valid = self.base | {'binaries': {'C3XRenderer.dll': 'dll'}}
        for fields in ({'returncode': 1}, {'sources_unchanged': False},
                       {'unit_inputs': {}}, {'preview_only': True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_build(valid | fields, 'dll', self.inputs)
