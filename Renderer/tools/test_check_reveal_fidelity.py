import unittest
from Renderer.tools.check_reveal_fidelity import assess, excursion


class RevealFidelityTests(unittest.TestCase):
    def test_static_detail_and_small_jpeg_noise_pass(self):
        times=[94+i*.1 for i in range(161)]
        samples=[bytes([20+i%3,100,230-i%3]) for i in range(161)]
        self.assertEqual(assess(times,excursion(samples)),'pass')

    def test_blur_and_missing_shadow_controls_fail(self):
        times=[94+i*.1 for i in range(161)]
        for bad in (bytes([120,120,120]),bytes([80,160,255])):
            samples=[bytes([20,100,230])]*161
            samples[85]=bad
            self.assertEqual(assess(times,excursion(samples)),'fail')

    def test_permanent_endpoint_change_is_not_a_transient(self):
        samples=[bytes([20,100,230])]*80+[bytes([40,125,245])]*81
        self.assertEqual(max(excursion(samples)),0)

    def test_missing_observations_and_camera_motion_are_incomplete(self):
        self.assertEqual(assess([],[]),'incomplete')
        self.assertEqual(assess([94,110],[0,0]),'incomplete')
        times=[94+i*.1 for i in range(161)]
        self.assertEqual(assess(times,[0]*161,camera_stable=False),'incomplete')
