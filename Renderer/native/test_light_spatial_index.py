"""Differential tests for conservative scene-light candidate lists and equations."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.city_fidelity.scene_lights import upgrade
from Renderer.lab.platform import windows_root

class SpatialCityLighting(unittest.TestCase):
    def test_candidates_and_final_illumination(self):
        run_cpp((Path(__file__).with_suffix('.cpp')).read_text().replace('"city_fidelity/', '"Renderer/native/city_fidelity/'))

    def test_gpu_equations_match_full_scan(self):
        path=Path(__file__).parent/'test_light_spatial_index_gpu.cpp'
        shader=str(windows_root()/'Renderer/native/city_fidelity/local_lights.hlsl').replace('\\','/')
        run_cpp(path.read_text().replace('CITY_SHADER_PATH',shader),timeout=120)

    def test_adapter_handles_active_and_generated_routes(self):
        directory=Path(__file__).parent/'city_fidelity'
        for name in ['terrain','mountain','objects','city','feature','rigid_feature','hydrology','resource_body','resource_shadow','local_lights']:
            source=(directory/(name+'.hlsl')).read_text()
            self.assertEqual(source,upgrade(source))
            self.assertIn('q8_candidate_index',source)
            self.assertIn('int j=indexed?',source)
        # Previously frozen scene-sized shaders must acquire the same adapter.
        source=(directory/'local_lights.hlsl').read_text()
        self.assertIn('Q8LocalGridInfo.w>.5',source)

if __name__=='__main__':unittest.main()
