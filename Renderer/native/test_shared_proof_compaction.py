"""Exact proof order and native caster staging admission with pinned owners."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_shared_instance_submission import GPU_STUB


ORDER_PROGRAM = GPU_STUB + r'''

#include <cstdio>
int main(){
 static_assert(Owner::budget==33554432 && Owner::record_limit==65536 && Owner::entry_limit==16384,"unchanged caps");
 ID3D11Device device;Owner owner;float projection[]={0,0,128,1260};Owner::Range range;
 std::array<std::shared_ptr<int>,3> sources={std::make_shared<int>(100),std::make_shared<int>(20),std::make_shared<int>(50)};
 std::array<Owner::Instance,3> values;unsigned keys[]={100,20,50};
 auto build=owner.begin_retained(Owner::Key{1});assert(build);
 for(unsigned n=0;n<3;++n){values[n].place[0]=float(keys[n]);
  assert(owner.append(build,Owner::Key{keys[n]},sources[n].get(),&values[n],1,projection,0,0,0,40,range));
  assert(range.first==n);
  Owner::RetainedSource proof;proof.source=sources[n];proof.owner={0,keys[n]};
  proof.canonical_source=Owner::Generation::source_key(sources[n].get(),40);proof.canonical=true;
  assert(owner.retain_source(build,Owner::Key{keys[n]},proof));
  assert(owner.retain_source(build,Owner::Key{keys[n]},proof));assert(build->retained_sources.size()==n+1);
 }
 // A range can remain unproved; full key differences never alias its proof.
 Owner::Key unproved{20};unproved[23]=1;
 assert(owner.append(build,unproved,sources[1].get(),&values[1],1,projection,0,0,0,40,range));
 auto old=owner.upload(build,&device);build.reset();assert(old && old->records==4);
 auto next=owner.begin_retained(Owner::Key{2});values[1].place[0]=99;
 assert(owner.append(next,Owner::Key{80},sources[1].get(),&values[1],1,projection,11,22,22,40,range));
 Owner::RetainedSource proof;proof.source=sources[1];proof.owner={0,20};
 proof.canonical_source=Owner::Generation::source_key(sources[1].get(),40);
 assert(!owner.retain_source(next,Owner::Key{999},proof));assert(owner.retain_source(next,Owner::Key{80},proof));
 std::vector<unsigned> order;
 assert(owner.carry_forward(next,[&](auto const& retained){order.push_back(unsigned(retained.owner[1]));return true;}));
 assert((order==std::vector<unsigned>{20,50,100}));
 auto current=owner.upload(next,&device);next.reset();assert(current && current->records==4);
 assert(current->find(Owner::Key{80}).first==0 && current->find(Owner::Key{20}).first==1);
 assert(current->find(Owner::Key{50}).first==2 && current->find(Owner::Key{100}).first==3 && !current->find(unproved));
 assert(current->source(sources[1].get(),40).first==0);
 Owner::Instance copied;std::memcpy(&copied,current->buffer->data.data()+64,64);assert(copied.place[0]==20);
 assert(old->records==4 && old->find(Owner::Key{100}).first==0);
 unsigned index=0;auto selected=owner.prepare_selection(&device,old,&index,1,128);assert(selected && selected->content==old);
 sources[2].reset();auto replacement=owner.begin_retained(Owner::Key{3});
 assert(owner.carry_forward(replacement,[](auto const& retained){return retained.owner[1]!=100;}));
 auto last=owner.upload(replacement,&device);replacement.reset();assert(last);
 assert(last->find(Owner::Key{20}) && last->find(Owner::Key{80}) && !last->find(Owner::Key{50}) && !last->find(Owner::Key{100}));
 assert(old->records==4 && old->buffer && selected->buffer);
 assert(owner.bytes()<=Owner::budget && owner.peak_bytes()<=Owner::budget);
 owner.clear();assert(owner.bytes()>0 && !owner.valid(old));old.reset();current.reset();last.reset();selected.reset();
 assert(!owner.bytes() && device.freed==device.creates);
}
'''


PRESSURE_PROGRAM = GPU_STUB + r'''

int main(){
 // This is the native generic camera's measured first caster admission:
 // old front stays pinned while 32764 next records cross the 32768 capacity.
 constexpr unsigned source_count=4764,old_ranges=15087,next_ranges=13860;
 constexpr unsigned old_records=36660,next_records=32764;
 constexpr std::size_t other_metadata_bytes=4428360;
 ID3D11Device device;Owner owner;float projection[]={0,0,128,1260};Owner::Range range;
 auto sources=std::make_shared<std::array<int,source_count>>();
 std::array<Owner::Instance,5> values;values[0].place[0]=31;
 auto fill=[&](Owner::Builder const& builder,unsigned prefix,unsigned count,unsigned records){
  auto three=records-2*count;
  for(unsigned n=0;n<count;++n){Owner::Key key{prefix,n};key[23]=n+1;
   auto source=&(*sources)[n%source_count];auto number=n<three?3u:2u;
   assert(owner.append(builder,key,source,values.data(),number,projection,0,0,0,40,range));
   Owner::RetainedSource proof;proof.source=sources;proof.owner={0,std::uint64_t(n%source_count)+1};
   proof.canonical_source=Owner::Generation::source_key(source,40);
   assert(owner.retain_source(builder,key,proof));
  }
  assert(builder->records==records && builder->ranges.size()==count && builder->retained_sources.size()==count);
 };
 auto old_builder=owner.begin_retained(Owner::Key{1});fill(old_builder,1,old_ranges,old_records);
 auto old=owner.upload(old_builder,&device);old_builder.reset();assert(old && old->placements.capacity()==65536);
 // Caller/group/scratch/source-handle storage remains jointly charged.
 auto other=owner.retain_metadata(other_metadata_bytes-sizeof(Owner::CpuAllocation));assert(other && other->bytes()==other_metadata_bytes);
 auto next=owner.begin_retained(Owner::Key{2});fill(next,2,next_ranges,next_records);
 assert(next->staging.capacity()==32768);
 auto appended=owner.append(next,Owner::Key{3},sources.get(),values.data(),5,projection,0,0,0,40,range);
 assert(appended);
 assert(next->records==32769 && next->staging.capacity()==65536 && old->records==old_records && old->buffer);
 auto current=owner.upload(next,&device);next.reset();assert(current);
 unsigned first=0;auto selected=owner.prepare_selection(&device,old,&first,1,128);assert(selected && selected->content==old);
 Owner::Instance actual;std::memcpy(&actual,old->buffer->data.data(),sizeof(actual));assert(actual.place[0]==31);
 assert(other->bytes()==other_metadata_bytes && owner.bytes()<=Owner::budget && owner.peak_bytes()<=Owner::budget);
 owner.clear();assert(owner.bytes()>0);selected.reset();old.reset();current.reset();other.reset();
 assert(!owner.bytes() && device.creates==device.freed);
}
'''


class SharedProofCompactionTests(unittest.TestCase):
    def test_exact_key_order_unproved_filtering_required_precedence_and_leases(self):
        run_cpp(ORDER_PROGRAM)

    def test_native_caster_capacity_crossing_keeps_old_generation_and_budget(self):
        run_cpp(PRESSURE_PROGRAM, timeout=60)


if __name__ == "__main__":
    unittest.main()
