"""Opt-in D3D11 test of replacement buffers and pinned reader retirement.

Run with C3X_RENDERER_GPU_TESTS=1 on Windows or the configured Windows VM.
Readback is confined to this fixture's buffers of at most 576 bytes.
"""
import os
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp


@unittest.skipUnless(os.environ.get("C3X_RENDERER_GPU_TESTS") == "1",
                     "set C3X_RENDERER_GPU_TESTS=1 for the native D3D11 fixture")
class SharedInstanceGpuTests(unittest.TestCase):
    def test_changed_upload_gpu_carry_and_pinned_front(self):
        run_cpp(Path(__file__).with_suffix(".cpp").read_text(), timeout=60)


if __name__ == "__main__":
    unittest.main()
