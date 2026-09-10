"""Execute the actual injected unit capture/forwarding functions with native mocks."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare():
    source = (ROOT/"injected_code.c").read_bytes().decode()
    start = source.index("bool\nforward_custom_unit_body")
    end = source.index("bool\nconfigure_custom_renderer_effects", start)
    output = ROOT/"Renderer/native/build/unit_bridge_capture.h"
    output.parent.mkdir(exist_ok=True)
    output.write_text(source[start:end])


class UnitBridgeTests(unittest.TestCase):
    def test_actual_capture_and_native_pass_through(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("portable C++ compiler unavailable")
        prepare()
        with tempfile.TemporaryDirectory(prefix="c3x-unit-bridge-") as scratch:
            binary = Path(scratch)/"bridge_test"
            subprocess.run([compiler, "-std=c++17", "-Werror",
                str(ROOT/"Renderer/native/test_unit_bridge.cpp"), "-o", str(binary)],
                check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
