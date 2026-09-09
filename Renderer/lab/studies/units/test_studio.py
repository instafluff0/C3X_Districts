import unittest
import numpy as np
from .studio import sample,raster,shadow_field,normalize,srgb,display

class StudioTests(unittest.TestCase):
    def test_srgb_roundtrip(self):
        values=np.linspace(0,1,1001);np.testing.assert_allclose(display(srgb(values)),values,atol=1e-7)

    def test_texel_centers_wrap_and_clamp(self):
        texture=np.array([[[0.],[1.]]]);uv=np.array([[.25,.5],[.75,.5],[1.25,.5]])
        np.testing.assert_allclose(sample(texture,uv)[:,0],[0,1,0]);np.testing.assert_allclose(sample(texture,uv,True)[:,0],[0,1,1])

    def test_triangle_barycentrics(self):
        points=np.array([[0.,0.,0.],[2.,0.,1.],[0.,2.,2.]])
        tri,x,y,b=next(raster(points,np.array([[0,1,2]]),3,3));i=np.where((x==0)&(y==0))[0][0]
        np.testing.assert_allclose(b[i],[.5,.25,.25]);self.assertAlmostEqual(float(b[i]@points[tri,2]),.75)

    def test_source_triangle_shadow_direction(self):
        p={'position':np.array([[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]]),'indices':np.array([[0,1,2],[0,2,3]])}
        visible=shadow_field([p],normalize(np.array([.2,.2,1.])),128)
        result=visible(np.array([[0,0,0],[0,0,1],[2,2,0]]))
        np.testing.assert_allclose(result,[0,1,1])

if __name__=='__main__':unittest.main()
