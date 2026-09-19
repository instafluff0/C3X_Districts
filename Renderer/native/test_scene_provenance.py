import unittest
from Renderer.native.native_cpp_test import run_cpp

class SceneProvenanceTests(unittest.TestCase):
    def test_copies_native_writes_lifetimes_and_bounds(self):
        run_cpp('int test_scene_provenance();int main(){return test_scene_provenance();}',
                sources=('Renderer/native/test_scene_provenance.cpp',))
