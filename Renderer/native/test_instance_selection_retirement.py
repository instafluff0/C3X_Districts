"""Host contracts for retiring completed instance plans before replacement."""
import os
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_shared_instance_submission import GPU_STUB


SOURCE = Path(__file__).resolve().parents[2] / "Renderer/sandbox/fresh_pipeline.h"


def production_retirement():
    source = SOURCE.read_text()
    start = source.index("        auto retire_completed_instance_plans=[&]{")
    end = source.index("        };", start) + len("        };")
    callback = source[start:end]
    start = source.index("bool prepare_instances(")
    end = source.index("    void batch_terrain_casters()", start)
    method = source[start:end]
    conditional = next(line.strip() for line in method.splitlines()
                       if "retire_completed_plans();" in line)
    return callback + "\n auto& retire_completed_plans=retire_completed_instance_plans;\n", conditional


PREAMBLE = GPU_STUB + r'''
#define C3X_RENDERER64_FRESH 1
#include "Renderer/native/render_core/frame_sample_cache.h"
struct Plan {Owner::SelectionLease selection;};
using Plans=c3x_renderer::render_core::FrameSampleCache<int,Plan,4>;
static_assert(Owner::budget==32u*1024u*1024u,"The joint allowance must remain unchanged");
'''


@unittest.skipIf(os.name == "nt", "Host-only contracts never dispatch Windows or VM tools")
class InstanceSelectionRetirementTests(unittest.TestCase):
    def test_completed_plans_retire_without_releasing_independent_consumers(self):
        callback, conditional = production_retirement()
        run_cpp(PREAMBLE + r'''
int main(){
 ID3D11Device device;ID3D11DeviceContext context{&device};Owner owner;Plans instance_plans;instance_plans.begin();
 std::size_t instance_plan_bytes=0;auto& plans=instance_plans;
 float projection[]={0,0,128,1260};Owner::Range range;
 std::vector<Owner::Instance> values(16384);
 auto builder=owner.begin_retained(Owner::Key{1});
 assert(owner.append(builder,Owner::Key{11},values.data(),values.data(),8192,projection,0,0,0,40,range));
 auto front=owner.upload(builder,&device);builder.reset();assert(front);
 auto pinned_front=front;auto old_bytes=front->buffer->data;
 std::weak_ptr<Owner::Generation const> old_generation=front;
 unsigned index=0;
 for(int key:{1,2}){
  auto slot=plans.select(key);
  plans[slot].value.selection=owner.prepare_selection(&device,front,&index,1,2u*1024u*1024u);
  plans[slot].valid=true;assert(plans[slot].value.selection);
  instance_plan_bytes+=plans[slot].value.selection->bytes();
 }
 // A borrowed consumer independently pins its selection and old generation.
 auto consumer=plans[0].value.selection;
 auto pressure=owner.retain_metadata(Owner::budget-owner.bytes()-256u*1024u);assert(pressure);
 builder=owner.begin_retained(Owner::Key{2},{},true);assert(builder);
 assert(!owner.append(builder,Owner::Key{12},values.data(),values.data(),unsigned(values.size()),projection,4,0,0,40,range));
 builder.reset();assert(owner.valid(front) && owner.valid(consumer));
 assert(front->buffer->data==old_bytes);
 auto before=owner.bytes();plans.begin();assert(owner.bytes()==before);
 // begin() advances the frame but does not release completed cache entries.
''' + callback + r'''
 auto reused=owner.find_covering([](auto const& candidate){return candidate.find(Owner::Key{12}).count==16384;});
 assert(!reused);
''' + conditional + r'''
 assert(plans.size()==0 && instance_plan_bytes==0 && owner.bytes()<before);
 assert(owner.valid(consumer) && owner.valid(pinned_front));
 assert(pinned_front->buffer->data==old_bytes);
 builder=owner.begin_retained(Owner::Key{2},{},true);assert(builder);
 assert(owner.append(builder,Owner::Key{12},values.data(),values.data(),unsigned(values.size()),projection,4,0,0,40,range));
 auto next=owner.upload(builder,&device,&context);builder.reset();assert(next);
 assert(next->find(Owner::Key{12}).count==values.size());
 assert(owner.valid(consumer) && owner.valid(pinned_front));
 assert(owner.bytes()<=Owner::budget && owner.peak_bytes()<=Owner::budget);
 // Successful replacement and release of ordinary old-front references do
 // not retire a generation still borrowed by an independent consumer.
 front.reset();pinned_front.reset();assert(!old_generation.expired());
 assert(consumer->content->buffer->data==old_bytes);
 auto pinned_bytes=owner.bytes();consumer.reset();assert(old_generation.expired());
 assert(owner.bytes()<pinned_bytes);
 owner.clear();assert(owner.bytes()>0);pressure.reset();next.reset();assert(!owner.bytes());
}
''')

    def test_unchanged_covering_union_preserves_plan_and_upload_reuse(self):
        callback, conditional = production_retirement()
        run_cpp(PREAMBLE + r'''
int main(){
 ID3D11Device device;Owner owner;Plans instance_plans;instance_plans.begin();
 auto& plans=instance_plans;std::size_t instance_plan_bytes=0;
 Owner::Instance value;float projection[]={0,0,128,1260};Owner::Range range;
 auto builder=owner.begin_retained(Owner::Key{1});
 assert(owner.append(builder,Owner::Key{11},&value,&value,1,projection,0,0,0,40,range));
 auto front=owner.upload(builder,&device);builder.reset();assert(front);
 unsigned index=0;auto slot=plans.select(1);
 plans[slot].value.selection=owner.prepare_selection(&device,front,&index,1,256);
 assert(plans[slot].value.selection);instance_plan_bytes=plans[slot].value.selection->bytes();
 auto bytes=owner.bytes(),plan_bytes=instance_plan_bytes;
 auto uploads=owner.uploads,plan_uploads=owner.plan_uploads,creates=device.creates;
 auto* selection=plans[slot].value.selection.get();
''' + callback + r'''
 for(unsigned frame=0;frame<8;++frame){
  plans.begin();
  auto reused=owner.find_covering([](auto const& candidate){return candidate.find(Owner::Key{11}).count==1;});
''' + conditional + r'''
  assert(reused==front && plans.size()==1 && plans[slot].value.selection.get()==selection);
  assert(owner.valid(plans[slot].value.selection));
  assert(owner.bytes()==bytes && instance_plan_bytes==plan_bytes);
  assert(owner.uploads==uploads && owner.plan_uploads==plan_uploads && device.creates==creates);
 }
 assert(owner.reuses==8);
 plans={};owner.clear();front.reset();assert(!owner.bytes());
}
''')

    def test_production_hook_precedes_replacement_charges_and_keeps_old_union(self):
        source = SOURCE.read_text()
        begin = source.index("bool prepare_instances(")
        end = source.index("    void batch_terrain_casters()", begin)
        method = source[begin:end]
        retire = method.index("if(!reused)retire_completed_plans();")
        self.assertLess(method.index("find_covering("), retire)
        for charge in ("reserve_index_scratch()", "retain_metadata(", "begin_retained("):
            self.assertLess(retire, method.index(charge))
        self.assertNotIn("shared_front.reset()", method)
        self.assertNotIn("shared_instances.clear()", method)
        self.assertIn("prepare_instances(inputs,retire_completed_plans,shared_front.get(),body_covered)", source)
        self.assertIn("body_inputs,retire_completed_instance_plans)", source)


if __name__ == "__main__":
    unittest.main()
