import unittest
from Renderer.native.native_cpp_test import run_cpp

class DirectUnitSceneTests(unittest.TestCase):
    def test_scene_samples_match_hardware_body_resolve(self):
        run_cpp('int test_gpu_unit_scene();int main(){return test_gpu_unit_scene();}',
                sources=('Renderer/native/test_gpu_unit_scene.cpp',),timeout=180)
