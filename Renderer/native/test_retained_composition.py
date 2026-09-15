import unittest
from Renderer.native.native_cpp_test import run_cpp

class RetainedCompositionTests(unittest.TestCase):
    def test_independent_visual_frames_and_native_versions(self):
        run_cpp('int test_retained_composition();int main(){return test_retained_composition();}',
                sources=('Renderer/native/test_retained_composition.cpp',),timeout=180)
