"""The outer gate must reject an incomplete native result, even with exit 0."""

from pathlib import Path
import struct
import tempfile
import unittest

from Renderer.native.helper_trial.run_gate1 import (
    MANDATORY_CHECKS, X86_MACHINE, X64_MACHINE, completed_exit, pe_machine,
    validate_native_report,
)


class GateReceiptTests(unittest.TestCase):
    @staticmethod
    def passing_report():
        return {
            "status": "pass",
            "presentation_backend": "NativePresenter-DComp",
            "cross_process_shared_import": True,
            "checks": {name: "pass" for name in MANDATORY_CHECKS},
            "desktop_witness": "unavailable",
            "timing": {"cpu_copy_submit_ms": 0.1},
            "memory": {"consumer_largest_free_bytes": 123},
            "adapter_luid": {"match": True},
        }

    def test_complete_native_receipt(self):
        self.assertEqual(validate_native_report(self.passing_report()), [])
        self.assertIn("presentation is unconfirmed",
                      validate_native_report(self.passing_report(), require_desktop=True)[0])
        report = self.passing_report() | {"desktop_witness": "pass"}
        self.assertEqual(validate_native_report(report, require_desktop=True), [])

    def test_unverified_boundary_never_passes(self):
        for changed in (
            {"presentation_backend": "standalone-HWND-swapchain"},
            {"cross_process_shared_import": False},
            {"checks": {"import": "pass"}},
            {"desktop_witness": "fail"},
            {"adapter_luid": {"match": False}},
            {"timing": {}},
            {"memory": {}},
        ):
            with self.subTest(changed=changed):
                report = self.passing_report() | changed
                self.assertTrue(validate_native_report(report))

    def test_only_matching_child_completion_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "completion.txt"
            self.assertIsNone(completed_exit(receipt, "current"))
            receipt.write_text("previous 0\n")
            self.assertIsNone(completed_exit(receipt, "current"))
            receipt.write_text("current not_an_exit\n")
            self.assertIsNone(completed_exit(receipt, "current"))
            receipt.write_text("current 3\n")
            self.assertEqual(completed_exit(receipt, "current"), 3)

    def test_pe_machine_header_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe.exe"
            image = bytearray(256)
            image[:2] = b"MZ"
            struct.pack_into("<I", image, 0x3C, 0x80)
            image[0x80:0x84] = b"PE\0\0"
            for machine in (X86_MACHINE, X64_MACHINE):
                struct.pack_into("<H", image, 0x84, machine)
                binary.write_bytes(image)
                self.assertEqual(pe_machine(binary), machine)
            binary.write_bytes(b"not an executable")
            with self.assertRaises(ValueError):
                pe_machine(binary)


if __name__ == "__main__":
    unittest.main()
