import unittest
from Renderer.tools.recover_city_auxiliary_uv import transfer
from Renderer.lab.shared.cities.auxiliary import restore
from Renderer.lab.shared.cities.fingerprint import geometry_digest
from Renderer.lab.studies.cities.trim_subsurface import trim_mesh
from Renderer.tools.asset_compiler.wall_mesh_filter import trim_skirt_and_ground

class AuxiliaryUV(unittest.TestCase):
    def test_clipped_mesh_recovers_interpolated_channels_without_geometry_change(self):
        source=dict(asset_id='fixture',skin=None,topology=dict(primitive='triangles',indices=[0,1,2]),
                    vertices=[dict(position=p,normal=[0,0,1],uv0=uv,uv1=uv,uv2=[x*2 for x in uv])
                    for p,uv in [([0,0,-1],[0,0]),([1,0,1],[1,0]),([0,1,1],[0,1])]])
        for clipped in [trim_mesh(source,0)[0],trim_skirt_and_ground(source,0)[0]]:
            derivative={**clipped,'vertices':[{k:v for k,v in p.items() if k not in ('uv1','uv2')} for p in clipped['vertices']]}
            recovered=restore(derivative,{geometry_digest(derivative):transfer(clipped,derivative)})
            self.assertEqual(recovered,clipped)
            for p in recovered['vertices']:
                if p['position']==[.5,0,0]:self.assertEqual(p['uv1'],[.5,0]);self.assertEqual(p['uv2'],[1,0])

if __name__=='__main__':unittest.main()
