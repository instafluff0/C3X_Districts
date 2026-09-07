"""Verify color-space and texture-coordinate inputs to derived light proxies."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header

spec=importlib.util.spec_from_file_location('facade_probe',Path(__file__).with_name('city_facade_light_probe.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class FacadeSampling(unittest.TestCase):
    def test_light_tracks_rotated_wall_instead_of_axis_aligned_box(self):
        normal=np.array([.5,3**.5/2,0]);tangent=np.array([-normal[1],normal[0],0])
        points=np.array([normal*.2+tangent*t+np.array([0,0,.1]) for t in [-.15,0,.15]])
        position,direction=probe.facade_plane_proxy(points,np.tile(normal,(3,1)),np.ones(3))
        np.testing.assert_allclose(direction,normal)
        np.testing.assert_allclose(position,normal*.212+np.array([0,0,.1]))
        self.assertAlmostEqual(float((position-points[1])@normal),.012)

    def test_facade_proxy_is_translation_and_rotation_equivariant(self):
        points=np.array([[.2,-.1,.1],[.2,.1,.1]]);normals=np.array([[1.,0,0],[1.,0,0]])
        rot=np.array([[.6,-.8,0],[.8,.6,0],[0,0,1]]);shift=np.array([3.,-2.,.4])
        p,n=probe.facade_plane_proxy(points,normals,np.array([1.,2.]))
        q,m=probe.facade_plane_proxy(points@rot.T+shift,normals@rot.T,np.array([1.,2.]))
        np.testing.assert_allclose(q,p@rot.T+shift);np.testing.assert_allclose(m,n@rot.T)

    def test_proxy_budget_keeps_dim_building_and_original_order(self):
        lights=[dict(owner=owner,intensity=intensity,color_linear=[1,1,1],range=1)
                for owner,intensity in [(0,9),(0,8),(0,7),(1,.1),(1,.2),(2,1)]]
        self.assertEqual(probe.bounded_lights(lights,4),[lights[i] for i in (0,1,4,5)])
        self.assertIs(probe.bounded_lights(lights,6),lights)
        with self.assertRaisesRegex(ValueError,'every emitting building'):
            probe.bounded_lights(lights,2)

    def test_srgb_midpoint_is_not_treated_as_linear_radiance(self):
        np.testing.assert_allclose(probe.linear(np.array([0,.5,1])),[0,.21404114048223255,1],rtol=1e-10)

    def test_dx10_srgb_alias_preserves_bc1_payload_and_source_file(self):
        payload=make_dds_dx10_header({'width':4,'height':4,'mip_count':1,'dxgi_format':72})+struct.pack('<HHI',0xf800,0,0)
        with tempfile.TemporaryDirectory(prefix='city-emission-dds-') as temporary:
            path=Path(temporary)/'emission.dds';path.write_bytes(payload)
            result=probe.emission_texture(path)
            np.testing.assert_array_equal(result,np.tile([1.,0.,0.],(4,4,1)))
            self.assertEqual(path.read_bytes(),payload)

    def test_half_texel_bilinear_clamp_matches_emission_sampler(self):
        image=np.array([[[1.,0.,0.],[0.,1.,0.]],[[0.,0.,1.],[1.,1.,1.]]])
        np.testing.assert_array_equal(probe.bilinear(image,[-1,-1]),[1,0,0])
        np.testing.assert_array_equal(probe.bilinear(image,[.25,.25]),[1,0,0])
        np.testing.assert_array_equal(probe.bilinear(image,[2,2]),[1,1,1])
        np.testing.assert_array_equal(probe.bilinear(image,[.5,.5]),[.5,.5,.5])


if __name__=='__main__':unittest.main()
