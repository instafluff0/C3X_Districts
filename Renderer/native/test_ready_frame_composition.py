"""Production pending-frame pixels, projected delivery, UI and retirement."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
class ReadyFrameCompositionTests(unittest.TestCase):
    def test_completed_front_survives_preparation_and_retirement(self):
        run_cpp('int test_ready_frame_composition();int main(){return test_ready_frame_composition();}',
                sources=('Renderer/native/test_ready_frame_composition.cpp',),timeout=180)
