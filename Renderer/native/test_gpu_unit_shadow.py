"""Independent CPU raster oracle for the production GPU unit shadow pass."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class GpuUnitShadowTests(unittest.TestCase):
    def test_selected_pass_and_finished_pixels(self):
        run_cpp('int test_gpu_unit_shadow(); int main(){return test_gpu_unit_shadow();}',
                sources=('Renderer/native/test_gpu_unit_shadow.cpp',), timeout=90)


if __name__ == '__main__':
    unittest.main()
