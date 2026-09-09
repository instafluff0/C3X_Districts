"""Execute the shared frame and actual unit shadow projection together."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class SceneLighting(unittest.TestCase):
    def test_world_and_unit_projection_and_normals(self):
        native = Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "lighting"
            subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(native / "test_scene_lighting.cpp"),
                            str(native / "environment_runtime.cpp"), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)

    def test_both_production_receivers_use_shared_filter(self):
        from Renderer.native.source_fidelity.prepare import ROOT, LAB, function
        shared = (LAB / "shaders/lighting/paged_shadow_v1.hlsl").read_text()
        for provider in ("source_fidelity/terrain.hlsl", "environment_refresh/feature.hlsl",
                         "city_fidelity/feature.hlsl"):
            shader = (ROOT / "Renderer/native" / provider).read_text()
            for name in ("pickup_page", "pickup_blocker", "c3x_paged_visibility"):
                self.assertEqual(function(shared, name), function(shader, name), provider)
