"""Consumers of the view read Civ III's capture, not the world window.

The world window is resident membership (resident world restructure, R1). The
camera job passed it to visibility coverage, arrival visibility, waves, unit
selection and the fresh draw, while display frames passed Civ III's capture,
so visibility coverage rebuilt after every step, and every one of them paid
for the whole window (performance review, section 60).
"""
import re
import unittest
from Renderer.lab.platform import ROOT


class ViewConsumersNativeFrameTests(unittest.TestCase):
    def test_camera_job_view_consumers_take_the_native_frame(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('    bool render(c3x_renderer_frame_v1 const & native_frame, c3x_renderer_output_v1 & output,')
        body = source[start:source.index('\n    }\n', source.index('frame_cache_path="resident-fresh";', start))]
        for call in ('arrival_visibility.capture(native_frame,', 'visibility_coverage.capture(native_frame)',
                     'prepare_wave_chunks(native_frame)', 'select_frame_units(native_frame,candidates,fresh_unit_poses',
                     'c3x_renderer64_render_fresh(native_frame,target)'):
            self.assertTrue(call in body, call)
        for call in ('arrival_visibility.capture(frame,', 'visibility_coverage.capture(frame)',
                     'prepare_wave_chunks(frame)', 'c3x_renderer64_render_fresh(frame,target)'):
            self.assertFalse(call in body, call)


if __name__ == '__main__':
    unittest.main()
