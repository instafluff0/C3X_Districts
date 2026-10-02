"""Opt-in exact spatial/interpreter D3D11 composition oracle.

Readback is test-only. Production composition retains GPU allocations.
"""
import os
import unittest
from Renderer.native.native_cpp_test import run_cpp


@unittest.skipUnless(os.environ.get("C3X_RENDERER_GPU_TESTS") == "1",
                     "set C3X_RENDERER_GPU_TESTS=1 for the native D3D11 fixture")
class SpatialCompositionTests(unittest.TestCase):
    def test_dense_ordered_pixels_source_boundaries_and_target_rebinding(self):
        run_cpp('int test_spatial_composition();int main(){return test_spatial_composition();}',
                sources=('Renderer/native/test_spatial_composition.cpp',),timeout=180)
