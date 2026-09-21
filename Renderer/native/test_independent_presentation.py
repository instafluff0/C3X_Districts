import unittest
from Renderer.native.native_cpp_test import run_cpp


class IndependentPresentationTests(unittest.TestCase):
    def test_present_and_restore_gdi_with_blocked_window_thread(self):
        run_cpp('int test_independent_presentation(); int main(){return test_independent_presentation();}',
                sources=('Renderer/native/test_independent_presentation.cpp',), timeout=30)
