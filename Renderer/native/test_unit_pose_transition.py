"""Real local-joint transition math, lifecycle boundaries, and rig serialization."""
import copy
import math
import tempfile
import unittest
from pathlib import Path

from Renderer.native.native_cpp_test import run_cpp
from Renderer.tools.asset_compiler import normalized_skin as skin
from Renderer.tools.asset_compiler.normalized_pose_cache import PoseCache
from Renderer.tools.asset_compiler.build_resource_animation_runtime import encode


class UnitPoseTransitionTests(unittest.TestCase):
    def test_limb_lengths_interruptions_facing_and_retirement(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_pose_transition.h"
#include "Renderer/native/unit_animation_runtime.h"
#include <cassert>
using namespace c3x_renderer;
using namespace c3x_renderer::render_core;
int main(){
 // The calibrated source forward vector projects onto the native compass.
 int dx[8]={1,1,1,0,-1,-1,-1,0},dy[8]={-1,0,1,1,1,0,-1,-1};
 for(int d=1;d<=8;++d){float a=native_unit_yaw(225,d);
  // Source forward is +X for this generic calibration proof.
  float x=std::cos(a)-std::sin(a),y=std::cos(a)+std::sin(a);
  assert(std::abs(x)<1e-5?dx[d-1]==0:x*dx[d-1]>0);
  assert(std::abs(y)<1e-5?dy[d-1]==0:y*dy[d-1]>0);
 }
 AnimationMesh mesh;mesh.duration=1;mesh.frames=3;mesh.bones=2;
 auto& r=mesh.rig;r.binding[0]=1;r.parents={-1,0};r.skin_joints={0,1};
 JointPose rest;rest.rotation={0,0,0,1};rest.scale={1,0,0,0,1,0,0,0,1};
 auto identity=joint_matrix(rest);r.inverse_bind={identity,identity};
 for(int f=0;f<3;++f){auto root=rest;float angle=float(f)*1.57079632679f;
  root.rotation={0,0,std::sin(angle/2),std::cos(angle/2)};
  auto hand=rest;hand.position={1,0,0};r.poses.push_back(root);r.poses.push_back(hand);}
 UnitPoseTransitions owner;
 assert(!owner.sample(7,1,1,0,1000,mesh,0));
 auto p=owner.sample(7,1,2,10,1000,mesh,1);assert(p&&std::abs(p[28]-1)<1e-6);
 p=owner.sample(7,1,2,70,1000,mesh,1);
 assert(p&&std::abs(p[28]-std::sqrt(.5))<1e-5&&std::abs(p[29]-std::sqrt(.5))<1e-5);
 assert(std::abs(std::hypot(p[28],p[29])-1)<1e-6); // elbow/hand distance never shrinks
 auto previous_x=p[28],previous_y=p[29];
 p=owner.sample(7,1,3,71,1000,mesh,2); // interrupted run -> attack, same visible pose
 assert(p&&std::abs(p[28]-previous_x)<1e-6&&std::abs(p[29]-previous_y)<1e-6);
 auto repeated=owner.sample(7,1,3,71,1000,mesh,2);assert(repeated==p); // reflection/body share pose
 for(int t=72;t<191;++t){p=owner.sample(7,1,3,t,1000,mesh,2);assert(p&&std::abs(std::hypot(p[28],p[29])-1)<1e-5);}
 assert(!owner.sample(7,1,3,191,1000,mesh,2)); // exact authored destination resumes
 int change_ticks=200;
 for(int action:{8,6,10,9,7,1}){ // fidget, death, capture, victory, fortify, idle
  assert(owner.sample(7,1,action,change_ticks++,1000,mesh,0));
 }
 assert(!owner.sample(7,2,1,500,1000,mesh,0)); // reused ID cannot blend an old corpse
 assert(owner.sample(7,2,8,501,1000,mesh,1));
 auto incompatible=mesh;incompatible.rig.binding[0]=2;
 assert(!owner.sample(7,2,6,502,1000,incompatible,2)); // bone count alone never authorizes mixing
 auto angle=owner.facing(7,2,0,1000,6.108652382f); // 350 degrees -> 10 degrees
 assert(std::abs(angle-6.108652382f)<1e-6);
 assert(std::abs(owner.facing(7,2,1,1000,.174532925f)-angle)<1e-6);
 assert(std::abs(owner.facing(7,2,38,1000,.174532925f)-6.283185307f)<.003);
 auto before=owner.facing(7,2,38,1000,.174532925f);
 assert(std::abs(owner.facing(7,2,39,1000,1.570796327f)-before)<1e-6);
 struct Visible{struct{int unit_id;}draw;std::uint64_t pose_identity;};
 owner.retain(std::vector<Visible>{});assert(owner.size()==0);
 assert(!owner.sample(7,2,6,600,1000,mesh,2)); // fog/reveal starts directly at current pose
 owner.clear();assert(owner.size()==0);
 // Locomotion arrival settles over 200 ms, including a held presentation
 // sample. All intermediate joints preserve the authored limb length.
 assert(!owner.sample(7,3,2,1000,1000,mesh,1));
 p=owner.sample(7,3,1,1010,1000,mesh,0);assert(p&&std::abs(p[29]-1)<1e-6);
 p=owner.sample(7,3,1,1010,1000,mesh,0);assert(p&&std::abs(p[29]-1)<1e-6);
 p=owner.sample(7,3,1,1110,1000,mesh,0);
 assert(p&&std::abs(p[28]-std::sqrt(.5))<1e-5&&std::abs(p[29]-std::sqrt(.5))<1e-5);
 assert(owner.sample(7,3,1,1209,1000,mesh,0));
 assert(!owner.sample(7,3,1,1210,1000,mesh,0));
}
''')

    def test_compiled_rig_matches_authored_palettes_and_rejects_bad_metadata(self):
        skeleton={"bones":[]}
        identity=[1.,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
        for i in range(3):
            skeleton["bones"].append({"name":f"joint{i}","parent":i-1,
                "inverse_bind_matrix":identity[:],"local":{"position":[0.,0.,0.],
                "orientation":[0.,0.,0.,1.],"scale_shear":[1.,0,0,0,1.,0,0,0,1.]}})
        mesh={"vertices":[{"position":[float(i==0),float(i==1),float(i==2)],
            "normal":[0.,0.,1.],"uv0":[0.,0.],"joints":[2,0,0,0],"weights":[1.,0,0,0]} for i in range(3)],
            "topology":{"indices":[0,1,2]}}
        matrices=[]
        for angle in (0.,math.pi*.7,math.pi):
            local=[copy.deepcopy(b["local"]) for b in skeleton["bones"]]
            local[0]["orientation"]=[0.,0.,math.sin(angle/2),math.cos(angle/2)]
            local[1]["position"]=[1.,2.,3.]
            local[1]["scale_shear"]=[1.,.2,0,.2,2.,0,0,0,-.5] # authored shear/reflection
            local[2]["scale_shear"]=[0.]*9 if angle==math.pi else [1.,0,0,0,1.,0,0,0,1.]
            matrices.extend(v for m in skin.world_matrices(skeleton,local) for v in m)
        cache=PoseCache(1.,2.,3,tuple(b["name"] for b in skeleton["bones"]),tuple(matrices))
        blob=encode(mesh,skeleton,cache,rig={"skeleton":skeleton,"cache":cache})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rig.bin';path.write_bytes(blob)
            run_cpp(r'''
#include "Renderer/native/render_core/unit_pose_transition.h"
#include <fstream>
#include <iterator>
#include <cassert>
using namespace c3x_renderer;using namespace c3x_renderer::render_core;
int main(){
 std::ifstream file("PATH",std::ios::binary);
 std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),{});
 AnimationMesh mesh;assert(decode_animation_mesh(bytes,mesh));
 auto const& r=mesh.rig;assert(r.parents.size()==3);
 for(unsigned frame=0;frame<mesh.frames;++frame){
  std::array<JointMatrix,256> worlds{};
  for(unsigned bone=0;bone<3;++bone){worlds[bone]=joint_matrix(r.poses[frame*3+bone]);
   if(r.parents[bone]>=0)worlds[bone]=joint_multiply(worlds[bone],worlds[r.parents[bone]]);
   auto palette=joint_multiply(r.inverse_bind[bone],worlds[r.skin_joints[bone]]);
   for(unsigned j=0;j<16;++j)assert(std::abs(palette[j]-mesh.palettes[(frame*3+bone)*16+j])<2e-5);
  }
 }
 auto old=mesh.vertices.size();bytes.back()=255;bytes.pop_back();assert(!decode_animation_mesh(bytes,mesh));
 assert(mesh.vertices.size()==old); // transactional reject
}
'''.replace('PATH',path.as_posix()))


if __name__=='__main__':unittest.main()
