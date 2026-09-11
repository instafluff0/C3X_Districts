"""Safety checks for maintenance, using disposable files rather than renderer workloads."""
import gzip
import os
from pathlib import Path
import tempfile
import time
import unittest

from Renderer.native.maintain_storage import ROOT, allowed, apply, snapshot


class StorageMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='storage-policy-', dir=ROOT / 'Renderer/native/build')
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'old.bmp'
        self.payload = b'BM' + bytes(8192)
        self.source.write_bytes(self.payload)
        old = time.time() - 3 * 86400
        os.utime(self.source, (old, old))
        self.record = {'schema': 1, 'minimum_age_days': 2, 'entries': [
            dict(path=self.source.relative_to(ROOT).as_posix(), action='gzip', **snapshot(self.source))]}

    def tearDown(self):
        self.temp.cleanup()

    def run_apply(self):
        return apply(self.record, self.directory / 'receipt.json')

    def test_archive_is_exact_and_journal_confirms_removal(self):
        result = self.run_apply()
        self.assertFalse(self.source.exists())
        with gzip.open(self.source.with_suffix('.bmp.gz'), 'rb') as stream:
            self.assertEqual(stream.read(), self.payload)
        self.assertTrue(result['results'][0]['removed'])

    def test_changed_source_is_preserved(self):
        self.source.write_bytes(b'new work')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.run_apply()
        self.assertEqual(self.source.read_bytes(), b'new work')

    def test_conflicting_archive_does_not_remove_source(self):
        self.source.with_suffix('.bmp.gz').write_bytes(gzip.compress(b'different'))
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.run_apply()
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_live_assets_and_linked_paths_are_excluded(self):
        self.assertFalse(allowed(ROOT / 'Renderer/packs/example.bmp'))
        self.assertFalse(allowed(ROOT / 'Renderer/lab/references/example.bmp'))
        self.assertFalse(allowed(ROOT / 'Renderer/native/build/candidate/example.bmp'))
        alias = self.directory / 'alias.bmp'
        alias.symlink_to(self.source)
        self.assertFalse(allowed(alias))

    def test_repeat_apply_cannot_erase_receipt(self):
        self.run_apply()
        receipt = self.directory / 'receipt.json'
        saved = receipt.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Receipt already exists'):
            self.run_apply()
        self.assertEqual(receipt.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
