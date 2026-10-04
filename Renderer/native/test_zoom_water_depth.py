"""GPU regression for animated water over resampled scenery depth."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class ZoomWaterDepthTests(unittest.TestCase):
    def test_water_plane_silhouettes_and_clear_depth_during_zoom(self):
        run_cpp((ROOT / 'Renderer/native/test_zoom_water_depth.cpp').read_text(), timeout=90)
