import unittest
from Renderer.native.native_cpp_test import run_cpp


class FullscreenHudRecipeTests(unittest.TestCase):
    def test_production_session_dense_native_hud(self):
        run_cpp('int test_fullscreen_hud_recipe();int main(){return test_fullscreen_hud_recipe();}',
                sources=('Renderer/native/test_fullscreen_hud_recipe.cpp',), timeout=240)
