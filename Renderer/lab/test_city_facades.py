"""Verify color-space and texture-coordinate inputs to derived light proxies."""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header

from Renderer.lab.shared.cities import facades as probe


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
            with patch.object(probe.city,'ROOT',Path(temporary).resolve()):
                result=probe.emission_texture(path)
            np.testing.assert_array_equal(result,np.tile([1.,0.,0.],(4,4,1)))
            self.assertEqual(path.read_bytes(),payload)

    def test_half_texel_bilinear_clamp_matches_emission_sampler(self):
        image=np.array([[[1.,0.,0.],[0.,1.,0.]],[[0.,0.,1.],[1.,1.,1.]]])
        np.testing.assert_array_equal(probe.bilinear(image,[-1,-1]),[1,0,0])
        np.testing.assert_array_equal(probe.bilinear(image,[.25,.25]),[1,0,0])
        np.testing.assert_array_equal(probe.bilinear(image,[2,2]),[1,1,1])
        np.testing.assert_array_equal(probe.bilinear(image,[.5,.5]),[.5,.5,.5])


class CityInputTracking(unittest.TestCase):
    def test_json_and_texture_bytes_share_one_read_closure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.city,'ROOT',Path(directory).resolve()):
            root=Path(directory).resolve()
            (root/'mesh.json').write_text('{"vertices":[]}')
            (root/'emission.dds').write_bytes(b'texture')
            (root/'generated').mkdir()
            (root/'generated/frames.json').write_text('{"meshes":{}}')
            with probe.city.track_inputs(generated=(root/'generated',)) as consumed:
                probe.city.read('mesh.json')
                probe.file_hash(root/'emission.dds')
                probe.city.read('generated/frames.json')
            self.assertEqual(set(consumed),{'mesh.json','emission.dds'})
            self.assertEqual(consumed['emission.dds'],probe.file_hash(root/'emission.dds'))

    def test_changed_bytes_within_one_build_fail_and_tracking_scope_recovers(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.city,'ROOT',Path(directory).resolve()):
            path=Path(directory)/'mesh.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed during build'):
                with probe.city.track_inputs():
                    probe.city.read('mesh.json')
                    path.write_text('{"new":true}')
                    probe.city.read('mesh.json')
            with probe.city.track_inputs() as consumed:
                self.assertEqual(probe.city.read('mesh.json'),{'new':True})
            self.assertEqual(set(consumed),{'mesh.json'})

    def test_nested_tracking_scopes_do_not_leak(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.city,'ROOT',Path(directory).resolve()):
            root=Path(directory)
            for name in ('outer.json','inner.json'):(root/name).write_text('{}')
            with probe.city.track_inputs() as outer:
                probe.city.read('outer.json')
                with probe.city.track_inputs() as inner:probe.city.read('inner.json')
            self.assertEqual(set(outer),{'outer.json'})
            self.assertEqual(set(inner),{'inner.json'})

    def test_pack_build_does_not_reuse_preceding_component_cache(self):
        from Renderer.native.city_fidelity import prepare_pack as builder
        with patch.object(builder,'_build',return_value={}), patch.object(probe.city.component,'cache_clear') as clear:
            builder.build_pack(builder.OUT)
            builder.build_pack(builder.OUT)
        self.assertEqual(clear.call_count,2)

    def test_output_cannot_overlap_preserved_city_inputs(self):
        from Renderer.native.city_fidelity import prepare_pack as builder
        with patch.object(builder,'_build') as build:
            for output in (builder.INPUT,builder.INPUT.parent,builder.ROOT/'Renderer/packs',
                           builder.ROOT/'Renderer/packs/CityStudyAuxiliaryUV/candidate'):
                with self.assertRaisesRegex(ValueError,'overlap'):
                    builder.build_pack(output)
        build.assert_not_called()


if __name__=='__main__':unittest.main()
