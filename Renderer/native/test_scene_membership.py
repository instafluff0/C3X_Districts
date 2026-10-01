"""Executable ownership contract for production's selected scene generations."""
import unittest
from Renderer.native.native_cpp_test import run_cpp

class SceneMembershipTests(unittest.TestCase):
    def test_camera_publication_borrows_and_mutations_retire_exact_generations(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_membership.h"
#include <array>
#include <cassert>
struct Chunk {struct Bounds {int left,top,right,bottom;} bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};};
struct Payload {unsigned* frees;~Payload(){++*frees;}};
using namespace c3x_renderer::render_core;
int main(){
 auto budget=std::make_shared<ResidentRetirement>();
 SceneMembership<Chunk,2> membership(budget);Chunk first;unsigned frees=0;
 auto mesh=std::make_shared<Payload>();mesh->frees=&frees;
 assert(membership.retain({1,100},mesh));membership.edit(0).push_back(first);
 auto old=membership.publish();auto same=membership.publish();
 assert(old==same && old->records[0].size()==1);
 auto version=old->revision;auto charge=budget->bytes.load();assert(charge>0);
 GeometryDrawView<Chunk,2> draw(membership);assert(draw.is(membership));
 // Read-only selection and traversal never create another generation.
 assert(membership[0].size()==1 && membership.revision()==version);
 assert(membership.publish()==same && budget->bytes==charge);
 // An explicit update detaches records and their shared content together.
 membership.edit(0)[0].translation_x=17;
 auto updated=membership.publish();assert(updated!=old && updated->revision>version);
 assert(old->records[0][0].translation_x==0 && updated->records[0][0].translation_x==17);
 assert(budget->bytes>charge && frees==0);
 membership.clear();mesh.reset();assert(frees==0);
 old.reset();same.reset();assert(frees==0);
 updated.reset();assert(frees==1 && budget->bytes==0);
}
''')

if __name__=='__main__':unittest.main()
