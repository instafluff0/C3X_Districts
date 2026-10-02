"""Independent canonical shadow pages, PCF continuity and attachment budget."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp


class ShadowSamplingGridTests(unittest.TestCase):
    def test_density_current_quality_rebase_pcf_guards_and_budget(self):
        source = Path(__file__).with_suffix(".cpp").read_text()
        run_cpp(source.replace('"render_core/', '"Renderer/native/render_core/'))


if __name__ == "__main__":
    unittest.main()
