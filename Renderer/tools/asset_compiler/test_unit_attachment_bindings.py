import unittest
import numpy as np
from .build_unit_animation_runtime import parent_skin_cache,inverse_transform
from .normalized_pose_cache import PoseCache

class Attachments(unittest.TestCase):
    def test_local_deformation_follows_socket_without_double_root_travel(self):
        roots=[];children=[];parents=[]
        for x in (3.,7.):
            root=np.eye(4);root[3,:3]=[x,2,1]
            joint=np.eye(4);joint[3,:3]=[.1*x,.2,.3]
            socket=np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[x,4,5,1.]])
            roots.append(root);children.extend([root,joint@root]);parents.append(socket)
        child=PoseCache(1,1,2,('Root','FlexibleTip'),tuple(np.array(children).reshape(-1)))
        parent=PoseCache(1,1,2,('Hand',),tuple(np.array(parents).reshape(-1)))
        result=np.array(parent_skin_cache(child,parent,'Hand','Root').matrices).reshape(2,2,4,4)
        for i in range(2):
            np.testing.assert_allclose(result[i,0],parents[i],atol=1e-12)
            np.testing.assert_allclose(result[i,1]@np.linalg.inv(result[i,0]),children[2*i+1]@np.linalg.inv(roots[i]),atol=1e-12)

    def test_nonrigid_root_inverse_and_singular_rejection(self):
        m=np.array([[2,.1,0,0],[0,3,.2,0],[0,0,.7,0],[7,8,9,1.]])
        np.testing.assert_allclose(np.array(inverse_transform(tuple(m.reshape(-1)))).reshape(4,4),np.linalg.inv(m),atol=1e-12)
        with self.assertRaisesRegex(ValueError,'singular'):inverse_transform((0.,)*16)

if __name__=='__main__':unittest.main()
