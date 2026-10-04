"""Executable ownership contract for production's selected scene generations."""
import unittest
from pathlib import Path
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
 // A unique generation can still have its revision observed by an index.
 // Actual removal advances that identity; retaining identical records does not.
 SceneMembership<Chunk,2> unique;Chunk source;GeometryDrawRecord<Chunk> entry(source);
 entry.tile_x=2;entry.owner={1,1};assert(unique.retain(entry.owner,std::make_shared<int>(7)));
 unique.edit(0).push_back(entry);auto prior=unique.revision();
 unique.retain_occurrences([](auto const&){return true;});assert(unique.revision()==prior);
 unique.retain_occurrences([](auto const&){return false;});assert(unique.revision()>prior && unique[0].empty());
 prior=unique.revision();unique.retain_occurrences([](auto const&){return false;});assert(unique.revision()==prior);
 unique.edit(0).push_back(entry);auto second=entry;second.tile_x=0;unique.edit(0).push_back(second);
 prior=unique.revision();auto order=unique.order_revision();
 assert(unique.order_occurrences([](auto const& draw){return draw.tile_x;}));
 assert(unique.revision()>prior && unique.order_revision()==order+1 && unique[0][0].tile_x==0);
 prior=unique.revision();order=unique.order_revision();
 assert(!unique.order_occurrences([](auto const& draw){return draw.tile_x;}));
 assert(unique.revision()==prior && unique.order_revision()==order);
 auto borrowed=unique.publish();
 assert(!unique.order_occurrences([](auto const& draw){return draw.tile_x;}) && unique.publish()==borrowed);
 assert(unique.order_occurrences([](auto const& draw){return -draw.tile_x;}));
 assert(unique.publish()!=borrowed && borrowed->records[0][0].tile_x==0 && unique[0][0].tile_x==2);
 order=unique.order_revision();unique.clear();assert(unique.order_revision()==order+1);
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

    def test_completed_fresh_selection_releases_old_meshes_before_next_view(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        begin = source.index('    void retire_geometry_selection(){')
        method = source[begin:source.index('    bool capture(', begin)]
        run_cpp(r'''#include "Renderer/native/render_core/scene_membership.h"
#include <cassert>
#include <array>
using namespace c3x_renderer::render_core;
struct Chunk {struct Bounds {int left,top,right,bottom;} bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};};
using Membership=SceneMembership<Chunk,2>;
struct Pass {
 Membership::Lease resident_lease;
 std::vector<int> resident,static_visible,water_visible,reflection_visible,all_visible;
 std::vector<int> roi_records,roi_shadow_records,selected_lighting,body_requirements;
 bool body_requirements_valid=true,visibility_valid=true;std::uint64_t resident_signature=42;
 struct Shadow {std::vector<int> casters,caster_inputs,instance_groups;Membership::Lease caster_lease;
  std::uint64_t caster_signature=42,prepared_signature=42;} shadow;
 std::array<int,4> completed_image{3,7,11,19};
''' + method + r'''};
int main(){
 auto budget=std::make_shared<ResidentRetirement>();Membership membership(budget);Pass pass;
 auto mesh=std::make_shared<int>(99);std::weak_ptr<int> weak=mesh;
 assert(membership.retain({1,1},mesh));
 pass.resident_lease=membership.publish();pass.shadow.caster_lease=membership.publish();
 // An independent in-flight consumer must still retain its generation.
 auto independent=membership.publish();mesh.reset();
 pass.body_requirements={1};pass.all_visible={2};pass.shadow.caster_inputs={3};
 pass.retire_geometry_selection();membership.clear();
 assert(!weak.expired());assert((pass.completed_image==std::array<int,4>({3,7,11,19})));
 assert(!pass.resident_lease && !pass.shadow.caster_lease && !pass.resident_signature);
 assert(!pass.visibility_valid && !pass.body_requirements_valid);
 assert(pass.body_requirements.empty() && pass.all_visible.empty() && pass.shadow.caster_inputs.empty());
 independent.reset();assert(weak.expired() && !budget->bytes);
 // The next scene uses the same bounded owner, with no accumulated leases.
 for(unsigned i=0;i<1000;++i){
  assert(membership.retain({1,i+2},std::make_shared<int>(i)));
  pass.resident_lease=membership.publish();pass.shadow.caster_lease=membership.publish();
  pass.retire_geometry_selection();membership.clear();assert(!budget->bytes);
 }
}
''')

if __name__=='__main__':unittest.main()
