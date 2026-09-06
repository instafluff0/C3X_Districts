import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'systems/sampling'))
from quality import box_linear,sharpen,srgb_decode,srgb_encode,uniform_transform,uv_metric,tangent_frame


class SamplingQuality(unittest.TestCase):
    def test_transfer_roundtrip_and_checker(self):
        x=np.linspace(0,1,256)
        np.testing.assert_allclose(srgb_encode(srgb_decode(x)),x,atol=1e-12)
        self.assertEqual(round(float(srgb_encode(.5))*255),188)

    def test_hdr_and_premultiplied_edge(self):
        rgba=np.array([[[8,2,1,1],[0,0,0,0]],[[8,2,1,1],[0,0,0,0]]])
        c,a=box_linear(rgba,np.array([[1,0],[1,0]]),2,[0,0,1,1])
        np.testing.assert_array_equal(c[0,0],[4,1,.5,.5]);self.assertEqual(a[0,0],.5)

    def test_invalid_hidden_color_and_overlay_boundary(self):
        x=np.ones((8,8,4));v=np.ones((8,8));v[:4,:4]=0;x[:4,:4]=[500,0,500,0]
        c,a=box_linear(x,v,2,[1,0,3,4])
        self.assertEqual(float(c[0,1].sum()),0)
        self.assertTrue((c[:,[0,3]]==0).all());self.assertTrue((a[:,[0,3]]==0).all())
        self.assertTrue((c[2,1]==1).all())

    def test_sharpen_negative_and_corrected_controls(self):
        x=np.zeros((16,32,3))+.1;x[:,16:]=.8
        bad=sharpen(x,bounded=False);good=sharpen(x,bounded=True)
        self.assertLess(bad.min(),.1);self.assertGreater(bad.max(),.8)
        self.assertGreaterEqual(good.min(),.1);self.assertLessEqual(good.max(),.8)

    def test_uniform_rotation_and_stretch_rejection(self):
        m=np.eye(4);m[:3,:3]=[[0,-2,0],[2,0,0],[0,0,2]];m[:3,3]=[3,4,5]
        self.assertAlmostEqual(uniform_transform(m),2)
        for matrix in [np.diag([2,1,1,1]),np.diag([-1,1,1,1]),np.diag([0,0,0,1])]:
            with self.assertRaises(ValueError):uniform_transform(matrix)

    def test_uv_stretch_density_and_tangent_handedness(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0]]);uv=np.array([[0,0],[1,0],[0,1]])
        self.assertAlmostEqual(uv_metric(p,uv)['anisotropy'],1)
        self.assertGreater(uv_metric(p,uv*[16,1])['anisotropy'],8)
        self.assertAlmostEqual(uv_metric(p*2,uv)['density'],.5)
        t,b,n,sign=tangent_frame(p,uv);np.testing.assert_array_equal([t,b,n],np.eye(3))
        self.assertEqual(sign,1);self.assertEqual(tangent_frame(p,uv*[-1,1])[3],-1)
        with self.assertRaises(ValueError):uv_metric(p,uv*0)

if __name__=='__main__':unittest.main()
