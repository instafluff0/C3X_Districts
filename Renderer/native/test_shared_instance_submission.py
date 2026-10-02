"""Executable persistent union ownership, upload and exact pass-input contracts."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


GPU_STUB = r'''
#include <cassert>
#include <cstring>
#include <vector>
#include <memory>
using UINT=unsigned;using HRESULT=int;
#define FAILED(value) ((value)<0)
enum {D3D11_USAGE_DEFAULT=0,D3D11_USAGE_IMMUTABLE=1,D3D11_BIND_VERTEX_BUFFER=2,D3D11_BIND_SHADER_RESOURCE=4,
 D3D11_RESOURCE_MISC_BUFFER_STRUCTURED=8,D3D11_USAGE_DYNAMIC=16,D3D11_CPU_ACCESS_WRITE=32,
 D3D11_MAP_WRITE_DISCARD=64,D3D11_MAP_WRITE_NO_OVERWRITE=128,DXGI_FORMAT_R32G32B32_FLOAT=1,
 DXGI_FORMAT_R32G32_FLOAT=2,DXGI_FORMAT_R32_UINT=3,D3D11_INPUT_PER_VERTEX_DATA=4,D3D11_INPUT_PER_INSTANCE_DATA=5};
struct D3D11_BUFFER_DESC {UINT ByteWidth=0,Usage=0,BindFlags=0,MiscFlags=0,StructureByteStride=0,CPUAccessFlags=0;};
struct D3D11_SUBRESOURCE_DATA {void const* pSysMem=nullptr;};
struct D3D11_MAPPED_SUBRESOURCE {void* pData=nullptr;};
struct D3D11_BOX {UINT left=0,top=0,front=0,right=0,bottom=0,back=0;};
struct ID3D11Buffer {std::vector<unsigned char> data;unsigned* freed;
 void Release(){++*freed;delete this;}};
struct ID3D11ShaderResourceView {void Release(){delete this;}};
struct ID3D11InputLayout {};
struct ID3DBlob {void* GetBufferPointer(){return nullptr;}unsigned GetBufferSize(){return 0;}};
struct D3D11_INPUT_ELEMENT_DESC {char const* name;UINT index,format,slot,offset,classification,step;};
struct ID3D11Device {bool fail=false,view_fail=false;unsigned creates=0,freed=0,refs=1;
 HRESULT CreateBuffer(D3D11_BUFFER_DESC const* d,D3D11_SUBRESOURCE_DATA const* initial,ID3D11Buffer** output){
  ++creates;if(fail)return -1;
  if(d->Usage==D3D11_USAGE_IMMUTABLE && d->BindFlags==D3D11_BIND_SHADER_RESOURCE)assert(d->StructureByteStride==64);
  if(d->Usage==D3D11_USAGE_IMMUTABLE && d->BindFlags==D3D11_BIND_VERTEX_BUFFER)assert(!d->StructureByteStride && initial);
  *output=new ID3D11Buffer;(*output)->freed=&freed;(*output)->data.resize(d->ByteWidth);
  if(initial)std::memcpy((*output)->data.data(),initial->pSysMem,d->ByteWidth);return 0;
 }
 HRESULT CreateShaderResourceView(ID3D11Buffer*,void*,ID3D11ShaderResourceView** output){if(view_fail)return -1;*output=new ID3D11ShaderResourceView;return 0;}
 HRESULT CreateInputLayout(D3D11_INPUT_ELEMENT_DESC const*,UINT,void*,UINT,ID3D11InputLayout**){return 0;}
 void Release(){--refs;}
};
struct ID3D11DeviceContext {ID3D11Device* device;bool fail=false;std::vector<UINT> modes;
 unsigned copies=0,updates=0;std::size_t copied_bytes=0,uploaded_bytes=0;
 void GetDevice(ID3D11Device** output){*output=device;++device->refs;}
 HRESULT Map(ID3D11Buffer* buffer,UINT,UINT mode,UINT,D3D11_MAPPED_SUBRESOURCE* mapped){if(fail)return -1;modes.push_back(mode);mapped->pData=buffer->data.data();return 0;}
 void Unmap(ID3D11Buffer*,UINT){}
 void CopySubresourceRegion(ID3D11Buffer* target,UINT,UINT x,UINT y,UINT z,ID3D11Buffer* source,UINT,D3D11_BOX const* box){
  assert(source!=target && !y && !z && box && box->top==0 && box->front==0 && box->bottom==1 && box->back==1);
  auto bytes=box->right-box->left;assert(x+bytes<=target->data.size() && box->right<=source->data.size());
  std::memcpy(target->data.data()+x,source->data.data()+box->left,bytes);++copies;copied_bytes+=bytes;
 }
 void UpdateSubresource(ID3D11Buffer* target,UINT,D3D11_BOX const* box,void const* data,UINT,UINT){
  assert(box && box->top==0 && box->front==0 && box->bottom==1 && box->back==1 && box->right<=target->data.size());
  auto bytes=box->right-box->left;std::memcpy(target->data.data()+box->left,data,bytes);++updates;uploaded_bytes+=bytes;
 }
};
#include "Renderer/native/render_core/shared_instance_submission.h"
using Owner=c3x_renderer::render_core::SharedInstanceSubmission;
'''


class SharedInstanceSubmissionTests(unittest.TestCase):
    def test_delta_replacement_packs_changed_only_and_preserves_indexed_old_bytes(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;ID3D11DeviceContext context{&device};Owner owner;Owner::Range range;
 float projection[]={17,23,128,1260};std::vector<Owner::Instance> values(3);
 auto a=std::make_shared<int>(11),b=std::make_shared<int>(12),c=std::make_shared<int>(13);
 auto proof=[](auto const& source,unsigned generation){Owner::RetainedSource value;value.source=source;
  value.owner={generation,generation};value.canonical_source=Owner::Generation::source_key(source.get(),40);return value;};
 auto pa=proof(a,11),pb=proof(b,12),pc=proof(c,13);
 auto builder=owner.begin_retained(Owner::Key{1},{},true);assert(builder);
 for(unsigned n=0;n<3;++n){values[0].place[0]=float(100+n);
  auto source=n==0?a:n==1?b:c;auto p=n==0?pa:n==1?pb:pc;
  assert(owner.append(builder,Owner::Key{11+n},source.get(),values.data(),3,projection,n*10,n*20,n*20,40,range));
  assert(owner.retain_source(builder,Owner::Key{11+n},p));
 }
 auto old=owner.upload(builder,&device,&context);builder.reset();assert(old && old->records==9);
 assert(old->placements.empty() && old->staging.empty() && owner.uploaded_bytes==9*64 && owner.packed_records==9);
 auto old_bytes=old->buffer->data;unsigned index=0;auto selected=owner.prepare_selection(&device,old,&index,1,64);assert(selected);
 builder=owner.begin_retained(Owner::Key{2},{},true);
 assert(owner.reuse_range(builder,Owner::Key{11},3,pa,range) && range.first==0 && builder->staging.empty());
 auto recycled=pa;++recycled.owner[1];assert(!owner.reuse_range(builder,Owner::Key{11},3,recycled,range));
 assert(!owner.reuse_range(builder,Owner::Key{12},2,pb,range));
 values[0].place[0]=999;
 assert(owner.append(builder,Owner::Key{22},b.get(),values.data(),3,projection,9,8,8,40,range));
 assert(owner.retain_source(builder,Owner::Key{22},pb));
 assert(owner.carry_forward(builder,[](auto const& source){return source.owner[1]==13;}));
 assert(builder->records==9 && builder->staging.size()==3 && builder->copies.size()==2);
 assert(owner.range_reuses==1 && owner.carried_ranges==1 && owner.packed_records==12);
 auto uploaded=owner.uploaded_bytes,copied=owner.copied_bytes;
 auto next=owner.upload(builder,&device,&context);builder.reset();assert(next);
 assert(owner.uploaded_bytes-uploaded==3*64 && owner.copied_bytes-copied==6*64 && owner.allocated_bytes==18*64);
 assert(context.copies==2 && context.copied_bytes==6*64 && context.uploaded_bytes==12*64);
 assert(next->buffer!=old->buffer && old->buffer->data==old_bytes && owner.valid(selected));
 auto next_a=next->find(Owner::Key{11}),next_b=next->find(Owner::Key{22}),next_c=next->find(Owner::Key{13});
 assert(!std::memcmp(next->buffer->data.data()+next_a.first*64,old_bytes.data(),3*64));
 assert(!std::memcmp(next->buffer->data.data()+next_c.first*64,old_bytes.data()+6*64,3*64));
 Owner::Instance changed;std::memcpy(&changed,next->buffer->data.data()+next_b.first*64,64);
 assert(changed.place[0]==999 && changed.view[0]==9 && changed.view[1]==8 && changed.view[3]==40);
 assert(next->placements.empty() && next->copies.empty() && next->updates.empty() && !next->copy_source);
 // Required ranges in a later replacement can reuse a generation that owns
 // GPU placements only, and the old indexed selection remains independent.
 builder=owner.begin_retained(Owner::Key{3},{},true);
 assert(owner.reuse_range(builder,Owner::Key{11},3,pa,range));
 auto operations=context.copies+context.updates;auto bytes=owner.bytes();device.view_fail=true;
 assert(!owner.upload(builder,&device,&context));device.view_fail=false;
 assert(context.copies+context.updates==operations && owner.select(Owner::Key{2})==next);
 assert(old->buffer->data==old_bytes && owner.bytes()<=bytes);
 ID3D11Device other_device;ID3D11DeviceContext other_context{&other_device};
 assert(!owner.upload(builder,&device,&other_context) && context.copies+context.updates==operations);
 auto third=owner.upload(builder,&device,&context);builder.reset();assert(third && third->records==3);
 assert(owner.valid(selected) && selected->content->buffer->data==old_bytes);
 assert(owner.bytes()<=Owner::budget && owner.peak_bytes()<=Owner::budget);
 owner.clear();assert(!owner.valid(selected) && owner.bytes()>0);
 next.reset();third.reset();old.reset();selected.reset();assert(!owner.bytes() && device.creates==device.freed);
}
''')

    def test_carried_source_fallback_survives_skipped_original_canonical_range(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;Owner owner;float projection[]={0,0,128,1260};Owner::Range range;
 auto source=std::make_shared<int>(7),required=std::make_shared<int>(9);
 std::vector<Owner::Instance> values(Owner::record_limit/2);
 Owner::RetainedSource proof;proof.source=source;proof.owner={0,9};
 proof.canonical_source=Owner::Generation::source_key(source.get(),40);
 auto builder=owner.begin_retained(Owner::Key{1});
 assert(owner.append(builder,Owner::Key{100},source.get(),values.data(),unsigned(values.size()),projection,0,0,0,40,range));
 proof.canonical=true;assert(owner.retain_source(builder,Owner::Key{100},proof));
 assert(owner.append(builder,Owner::Key{1},source.get(),values.data(),unsigned(values.size()),projection,128,64,64,40,range));
 proof.canonical=false;assert(owner.retain_source(builder,Owner::Key{1},proof));
 auto old=owner.upload(builder,&device);builder.reset();assert(old && old->records==Owner::record_limit);
 builder=owner.begin_retained(Owner::Key{2});
 assert(owner.append(builder,Owner::Key{200},required.get(),values.data(),1,projection,0,0,0,40,range));
 assert(owner.carry_forward(builder,[](auto const&){return true;}));
 auto next=owner.upload(builder,&device);builder.reset();assert(next);
 assert(next->find(Owner::Key{1}).count==values.size() && !next->find(Owner::Key{100}));
 // A validated carried wrapped range remains a canonical caster source even
 // when the original canonical range could not fit. Current required source
 // entries still take precedence because they were appended first.
 auto fallback=next->source(source.get(),40,unsigned(values.size()));
 assert(fallback.count==values.size() && fallback.first==next->find(Owner::Key{1}).first);
 assert(next->source(required.get(),40,1).first==next->find(Owner::Key{200}).first);
 assert(owner.bytes()<=Owner::budget && owner.peak_bytes()<=Owner::budget);
 owner.clear();old.reset();next.reset();assert(!owner.bytes());
}
''')

    def test_prepared_union_budget_required_priority_and_pinned_overlap(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;Owner owner;float projection[]={0,0,128,1260};Owner::Range range;
 auto source=std::make_shared<int>(7);Owner::RetainedSource proof;proof.source=source;proof.owner={0,9};
 proof.canonical_source=Owner::Generation::source_key(source.get(),40);proof.canonical=true;
 std::vector<Owner::Instance> values(Owner::record_limit/2);values[0].place[0]=31;
 auto first=owner.begin_retained(Owner::Key{1},source);
 assert(owner.append(first,Owner::Key{1},source.get(),values.data(),unsigned(values.size()),projection,0,0,0,40,range));
 assert(owner.retain_source(first,Owner::Key{1},proof));auto front=owner.upload(first,&device);first.reset();assert(front);
 assert(front->staging.empty() && front->placements.size()==front->records);
 assert(front->bytes()>=front->placements.capacity()*64+front->gpu_bytes());
 auto pinned=front;unsigned first_index=0;auto selected=owner.prepare_selection(&device,pinned,&first_index,1,16384);assert(selected);
 // Required ranges claim the canonical source first; carried ranges do not
 // replace it, including a recycled source identity with a fresh owner key.
 auto next=owner.begin_retained(Owner::Key{2},source);values[0].place[0]=89;
 assert(owner.append(next,Owner::Key{2},source.get(),values.data(),1,projection,111,222,222,40,range));
 assert(owner.retain_source(next,Owner::Key{2},proof));
 assert(owner.carry_forward(next,[](auto const&){return true;}));
 auto replacement=owner.upload(next,&device);next.reset();assert(replacement);
 assert(replacement->source(source.get(),40).first==replacement->find(Owner::Key{2}).first);
 auto packed=reinterpret_cast<Owner::Instance const*>(replacement->buffer->data.data());
 assert(packed[replacement->find(Owner::Key{2}).first].place[0]==89);
 if(auto carried=replacement->find(Owner::Key{1}))assert(packed[carried.first].place[0]==31);
 assert(owner.bytes()<=Owner::budget && owner.peak_bytes()<=Owner::budget);
 // Allocation and view failure keep the published front and copied ranges.
 auto fail=owner.begin_retained(Owner::Key{3});
 assert(owner.append(fail,Owner::Key{3},source.get(),values.data(),1,projection,0,0,0,40,range));
 device.fail=true;assert(!owner.upload(fail,&device));device.fail=false;
 assert(owner.select(Owner::Key{2})==replacement);fail.reset();
 auto pressure=owner.retain_metadata(Owner::budget-owner.bytes()-256);assert(pressure);
 auto bounded=owner.begin_retained(Owner::Key{4});
 if(bounded){assert(!owner.append(bounded,Owner::Key{4},source.get(),values.data(),unsigned(values.size()),projection,0,0,0,40,range));}
 assert(owner.bytes()<=Owner::budget);bounded.reset();pressure.reset();
 owner.clear();assert(!owner.valid(replacement) && !owner.valid(pinned) && owner.bytes()>0);
 replacement.reset();front.reset();pinned.reset();selected.reset();assert(!owner.bytes());
}
''')

    def test_legacy_selected_batch_keeps_exact_owner_and_canonical_source_count(self):
        cpp = (ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        key_start = cpp.index('    c3x_renderer::render_core::SharedInstanceSubmission::Key shared_instance_draw_key(')
        key_end = cpp.index('\n    }', key_start) + len('\n    }')
        key = cpp[key_start:key_end]
        selected_start = cpp.index('                    GeometryDrawRecord record', cpp.index('    bool submit_scene_pass('))
        selected_end = cpp.index('                    receivers.clear();', selected_start)
        selection = cpp[selected_start:selected_end]
        run_cpp(GPU_STUB + r'''
#include "Renderer/native/render_core/geometry_draws.h"
struct Mesh {
 std::array<int,4> bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};
 std::uint64_t version=0;std::shared_ptr<std::vector<Owner::Instance> const> instances;float instance_material=40;
};
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Mesh,2>;
using GeometryDrawReference=GeometryDrawView::Reference;
using GeometryDrawRecord=GeometryDrawView::Record;
struct Harness {
''' + key + r'''
 GeometryDrawRecord select(GeometryDrawReference const& item){
''' + selection + r'''
  return record;
 }
};
int main(){
 ID3D11Device device;Owner owner;Harness harness;Mesh mesh;
 mesh.instances=std::make_shared<std::vector<Owner::Instance>>(3);mesh.version=89;mesh.instance_material=21.18f;
 GeometryDrawRecord occurrence(mesh);occurrence.owner={3,712};occurrence.ordinal=47;
 occurrence.translation_x=-771;occurrence.translation_y=281;
 float projection[]={17,23,128,1260};std::memcpy(occurrence.natural_projection,projection,sizeof(projection));
 GeometryDrawReference input(occurrence);auto original=harness.shared_instance_draw_key(1,input);
 auto builder=owner.begin(Owner::Key{1});Owner::Range range;
 assert(owner.append(builder,original,mesh.instances.get(),mesh.instances->data(),3,projection,-771,281,281,21.18f,range));
 auto front=owner.upload(builder,&device);builder.reset();assert(front);
 // Reconstructing only content used generation zero and missed the union.
 GeometryDrawRecord dropped(mesh);dropped.translation_x=-771;dropped.translation_y=281;
 std::memcpy(dropped.natural_projection,projection,sizeof(projection));
 assert(!front->find(harness.shared_instance_draw_key(1,GeometryDrawReference(dropped))));
 // Execute the production selected-batch construction, including exact facts.
 auto selected=harness.select(input);assert(selected.owner==occurrence.owner);
 assert(front->find(harness.shared_instance_draw_key(1,GeometryDrawReference(selected))).count==3);
 ++selected.ordinal;assert(harness.shared_instance_draw_key(1,GeometryDrawReference(selected))==original);
 ++selected.owner.generation;assert(!front->find(harness.shared_instance_draw_key(1,GeometryDrawReference(selected))));
 assert(front->source(mesh.instances.get(),21.18f,3).count==3);
 assert(!front->source(mesh.instances.get(),21.18f,2));
 assert(!front->source(mesh.instances.get(),21.19f,3));
 assert(!front->source(&mesh,21.18f,3));
 owner.clear();front.reset();assert(!owner.bytes());
}
''')

    def test_stable_reassembly_lighting_and_subset_selection_reuse_exact_ranges(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;ID3D11DeviceContext context{&device};Owner owner;Owner::Instance placement;float projection[]={0,0,128,1260};Owner::Range body,shadow,duplicate;
 Owner::Key body_key={31,8ull<<32,51,64,72,4096,112,1};
 Owner::Key shadow_key={31,1,8,2,3,4,5,6,7,8,9,10,11,12,13,14,15,51,40,1};shadow_key.back()=1;
 auto builder=owner.begin(Owner::Key{10,2,3});
 assert(owner.append(builder,body_key,&placement,&placement,1,projection,64,72,72,40,body));
 assert(owner.append(builder,body_key,&placement,&placement,1,projection,64,72,72,40,duplicate));
 assert(duplicate.first==body.first && builder->records==1);
 assert(owner.append(builder,shadow_key,&placement,&placement,1,projection,0,0,0,40,shadow));
 auto front=owner.upload(builder,&device);builder.reset();assert(front);
 auto proof=[&](Owner::Generation const& candidate){return candidate.find(body_key).count==1 && candidate.find(shadow_key).count==1;};
 // Camera traversal ordinal, selected order and light/page constants are
 // outside immutable placement keys. A new view/lighting proof reuses GPU data.
 for(unsigned presentation=0;presentation<8;++presentation){
  assert(!owner.select(Owner::Key{100+presentation,2,3}));
  assert(owner.find_covering(proof)==front && owner.uploads==1 && device.creates==1);
 }
 assert(owner.find_covering([&](auto const& candidate){return candidate.find(body_key).count==1;})==front);
 unsigned repeated_order[]={body.first,shadow.first,body.first};
 assert(owner.select_indices(&device,&context,front,repeated_order,3));
 assert(!std::memcmp(owner.selection_buffer->data.data(),repeated_order,sizeof(repeated_order)));
 // Full caster facts, projection and native anchor changes cannot collide.
 for(unsigned changed:{3u,5u,18u,19u,23u}){auto altered=shadow_key;++altered[changed];
  assert(!owner.find_covering([&](auto const& candidate){return candidate.find(altered).count==1;}));}
 auto moved=body_key;++moved[5];assert(!owner.find_covering([&](auto const& candidate){return candidate.find(moved).count==1;}));
 assert(owner.uploads==1);owner.clear();front.reset();assert(!owner.bytes());
}
''')

    def test_empty_generation_and_foreign_owner_device_are_strict(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device,other_device;Owner owner,foreign;Owner::Instance placement;float projection[]={0,0,128,1260};Owner::Range range;
 auto empty_builder=owner.begin(Owner::Key{1});auto empty=owner.upload(empty_builder,&device);empty_builder.reset();
 assert(empty && owner.valid(empty) && !empty->records && !empty->buffer && device.creates==0);
 auto builder=owner.begin(Owner::Key{2});assert(!foreign.append(builder,Owner::Key{3},&placement,&placement,1,projection,0,0,0,40,range));
 assert(!foreign.upload(builder,&device));assert(owner.append(builder,Owner::Key{3},&placement,&placement,1,projection,0,0,0,40,range));
 assert(!owner.upload(builder,&other_device));auto lease=owner.upload(builder,&device);builder.reset();unsigned index=0;
 assert(owner.valid(lease) && !foreign.valid(lease) && !foreign.prepare_selection(&device,lease,&index,1));
 ID3D11DeviceContext wrong_context{&other_device};
 assert(!owner.select_indices(nullptr,&wrong_context,lease,&index,1) && !owner.selection_buffer && other_device.refs==1);
 auto metadata=owner.retain_metadata(1024);assert(metadata && metadata->bytes()>1024);
 owner.clear();empty.reset();lease.reset();assert(owner.bytes()==metadata->bytes());metadata.reset();assert(!owner.bytes());
}
''')

    def test_source_pins_charge_retired_assets_and_bound_replacement_staging(self):
        run_cpp(GPU_STUB + r'''
int main(){
 using Sources=c3x_renderer::render_core::SharedSourceResidency;Sources sources;
 auto full=Sources::gpu_budget;
 assert(sources.reserve(full/2,full));auto pin=sources.pin();
 assert(sources.gpu_bytes()==full && sources.bytes()>full+full/2);
 assert(!sources.reserve(full+1,full));assert(sources.gpu_bytes()==full);
 assert(sources.reserve(0,full));sources.clear();assert(sources.gpu_bytes()==full);
 // A replacement cannot silently hide its pinned predecessor's GPU bytes.
 assert(!sources.reserve(64,64));assert(sources.gpu_bytes()==full && sources.bytes()<=Sources::budget);
 pin.reset();assert(!sources.gpu_bytes());assert(sources.reserve(64,64));
 auto replacement=sources.pin();sources.clear();assert(sources.gpu_bytes()==64);
 replacement.reset();assert(!sources.bytes() && sources.peak_bytes()<=Sources::budget);
}
''')

    def test_immutable_selection_plan_lease_failure_budget_and_order(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;Owner owner;Owner::Instance placement;float projection[]={0,0,128,1260};Owner::Range range;
 auto builder=owner.begin(Owner::Key{1});std::vector<Owner::Instance> placements(3);
 assert(owner.append(builder,Owner::Key{2},&placement,placements.data(),3,projection,0,0,0,40,range));
 auto union_lease=owner.upload(builder,&device);builder.reset();unsigned order[]={2,0,2,1};
 auto creates=device.creates;unsigned invalid[]={3};
 assert(!owner.prepare_selection(&device,union_lease,invalid,1) && device.creates==creates);
 auto previous=owner.bytes();device.fail=true;
 assert(!owner.prepare_selection(&device,union_lease,order,4,8192) && owner.bytes()==previous);
 device.fail=false;auto plan=owner.prepare_selection(&device,union_lease,order,4,8192);
 assert(plan && owner.valid(plan) && plan->count==4 && plan->content==union_lease);
 assert(!std::memcmp(plan->buffer->data.data(),order,sizeof(order)) && plan->bytes()>8192+sizeof(order));
 assert(owner.plan_uploads==1 && owner.plan_uploaded_bytes==sizeof(order));
 assert(!owner.prepare_selection(&device,union_lease,order,4,Owner::budget) && owner.valid(plan));
 std::vector<Owner::SelectionLease> pressure;bool rejected=false;
 for(unsigned n=0;n<40;++n){auto next=owner.prepare_selection(&device,union_lease,order,4,1024u*1024u);
  if(!next){rejected=true;break;}pressure.push_back(next);assert(owner.bytes()<=Owner::budget);}
 assert(rejected && pressure.size()>20 && owner.peak_bytes()<=Owner::budget);
 owner.clear();assert(!owner.valid(plan) && owner.bytes()>0);union_lease.reset();
 assert(plan->content->records==3);pressure.clear();assert(owner.bytes()==plan->bytes()+plan->content->bytes());
 plan.reset();assert(!owner.bytes() && device.freed==device.creates-1);
}
''')

    def test_shader_adapters_flatten_resident_input_and_pin_source(self):
        from pathlib import Path
        from unittest.mock import patch
        import json
        import hashlib
        from Renderer.native.source_fidelity import prepare as natural
        from Renderer.native.environment_refresh import prepare as environment
        from Renderer.native.city_fidelity import prepare_shader as city
        produced = {}
        original_text, original_bytes = Path.read_text, Path.read_bytes
        def read_text(path, *args, **kwargs):
            return produced[path] if path in produced else original_text(path, *args, **kwargs)
        def read_bytes(path):
            return produced[path].encode() if path in produced else original_bytes(path)
        def write_text(path, content, *args, **kwargs):
            produced[path] = content
            return len(content)
        with patch.object(Path, 'read_text', read_text), patch.object(Path, 'read_bytes', read_bytes), patch.object(Path, 'write_text', write_text):
            natural.shaders()
            environment.main()
            city.main()
        for name in ['source_fidelity/objects.hlsl', 'source_fidelity/instance_caster.hlsl',
                     'environment_refresh/objects.hlsl', 'city_fidelity/objects.hlsl',
                     'city_fidelity/rigid_feature.hlsl', 'city_fidelity/rigid_caster.hlsl']:
            shader = produced[ROOT/'Renderer/native'/name]
            self.assertIn('StructuredBuffer<ResidentPlacement>', shader)
            self.assertNotIn('#include', shader)
        reflection = produced[ROOT/'Renderer/native/environment_refresh/objects.hlsl']
        self.assertIn('VSResidentReflectionInstance', reflection)
        caster = produced[ROOT/'Renderer/native/city_fidelity/rigid_caster.hlsl']
        self.assertIn('VSResidentPlacedCaster', caster)
        provenance = json.loads(produced[ROOT/'Renderer/native/city_fidelity/shader-provenance.json'])
        source = ROOT/'Renderer/lab/shared/shaders/objects/resident_instance.hlsl'
        self.assertEqual(provenance['source_sha256'][source.relative_to(ROOT).as_posix()], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_opaque_grouping_preserves_main_reflection_ties_and_barriers(self):
        run_cpp(GPU_STUB + r'''
struct Draw {unsigned id,mesh;bool opaque=true;int main_low,main_high,reflection_low,reflection_high;};
int main(){
 assert(c3x_renderer::render_core::opaque_rigid_material(21.18f));
 assert(!c3x_renderer::render_core::opaque_rigid_material(21.3f));
 assert(!c3x_renderer::render_core::opaque_rigid_material(60.f));
 assert(c3x_renderer::render_core::opaque_rigid_material(41.f));
 auto compatible=[](auto const& a,auto const& b){return a.mesh==b.mesh;};
 auto opaque=[](auto const& a){return a.opaque;};
 auto conflict=[](auto const& a,auto const& b){
  return !(a.main_high<b.main_low || a.main_low>b.main_high) ||
         !(a.reflection_high<b.reflection_low || a.reflection_low>b.reflection_high);
 };
 std::vector<Draw> separate={{0,1,true,0,2,0,2},{1,2,true,10,12,10,12},{2,1,true,20,22,20,22}};
 assert(c3x_renderer::render_core::group_independent_opaque(separate,compatible,opaque,conflict)==1);
 assert(separate[0].id==0 && separate[1].id==2 && separate[2].id==1);
 // LESS_EQUAL depth ties and source opacity cannot prove independent work.
 for(unsigned kind=0;kind<3;++kind){
  std::vector<Draw> blocked={{0,1,true,0,2,0,2},{1,2,true,10,12,10,12},{2,1,true,20,22,20,22}};
  if(kind==0){blocked[2].main_low=12;blocked[2].main_high=16;}
  if(kind==1){blocked[2].reflection_low=12;blocked[2].reflection_high=16;}
  if(kind==2)blocked[1].opaque=false;
  assert(c3x_renderer::render_core::group_independent_opaque(blocked,compatible,opaque,conflict)==0);
  for(unsigned i=0;i<3;++i)assert(blocked[i].id==i);
 }
 std::vector<Draw> bounded={{0,1,true,0,2,0,2},{1,2,true,10,12,10,12},{2,1,true,20,22,20,22}};
 assert(c3x_renderer::render_core::group_independent_opaque(bounded,compatible,opaque,conflict,1)==0);
 for(unsigned i=0;i<3;++i)assert(bounded[i].id==i);
}
''')

    def test_selected_indices_order_wrap_failure_and_device_ownership(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;ID3D11DeviceContext context{&device};Owner owner;
 Owner::Key identity={1},key={1};Owner::Range range;float projection[]={0,0,128,1260};
 std::vector<Owner::Instance> placements(4);auto builder=owner.begin(identity);
 assert(owner.append(builder,key,placements.data(),placements.data(),4,projection,0,0,0,40,range));
 device.view_fail=true;assert(!owner.upload(builder,&device) && !builder->buffer && !builder->view && builder->gpu_bytes()==0);
 device.view_fail=false;auto lease=owner.upload(builder,&device);assert(lease);builder.reset();
 unsigned invalid[]={4};auto creates=device.creates;
 assert(!owner.select_indices(&device,&context,lease,invalid,1) && device.creates==creates);
 unsigned order[]={2,0,2,1};assert(owner.select_indices(nullptr,&context,lease,order,4));
 assert(device.refs==1 && owner.selection_offset==0 && owner.selection_cursor==4);
 assert(context.modes.back()==D3D11_MAP_WRITE_DISCARD && owner.selection_uploaded_bytes==16);
 assert(!std::memcmp(owner.selection_buffer->data.data(),order,sizeof(order)));
 unsigned body_then_decal[]={3,0,1,2};assert(owner.select_indices(nullptr,&context,lease,body_then_decal,4));
 assert(owner.selection_offset==16 && context.modes.back()==D3D11_MAP_WRITE_NO_OVERWRITE);
 assert(!std::memcmp(owner.selection_buffer->data.data(),order,sizeof(order)));
 assert(!std::memcmp(owner.selection_buffer->data.data()+16,body_then_decal,sizeof(body_then_decal)));
 auto offset=owner.selection_offset,cursor=owner.selection_cursor;auto uploads=owner.selection_uploads;
 context.fail=true;assert(!owner.select_indices(&device,&context,lease,order,4));
 assert(owner.selection_offset==offset && owner.selection_cursor==cursor && owner.selection_uploads==uploads);context.fail=false;
 std::vector<unsigned> full(Owner::record_limit,2);assert(owner.select_indices(&device,&context,lease,full.data(),unsigned(full.size())));
 assert(owner.selection_offset==0 && owner.selection_cursor==Owner::record_limit && context.modes.back()==D3D11_MAP_WRITE_DISCARD);
 assert(owner.select_indices(&device,&context,lease,order,4) && owner.selection_offset==0);
 assert(owner.cpu_bytes()+owner.gpu_bytes()==owner.bytes() && owner.bytes()<=Owner::budget);
 // The selected stream retires immediately; pinned immutable consumers stay.
 auto unfinished=owner.begin(Owner::Key{2});
 owner.clear();assert(owner.bytes()==lease->bytes()+unfinished->bytes() && device.refs==1);
 assert(!owner.select_indices(&device,&context,lease,order,4));
 assert(!owner.upload(unfinished,&device));unfinished.reset();lease.reset();assert(!owner.bytes());
}
''')

    def test_union_ranges_exact_payload_reuse_and_pinned_retirement(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;Owner owner;Owner::Key identity={1,2,3};
 auto source=std::make_shared<std::vector<Owner::Instance>>(3);
 for(unsigned i=0;i<source->size();++i)for(unsigned n=0;n<8;++n)(*source)[i].place[n]=float(i*8+n)*.125f;
 float projection[]={-17,139,192,1260};auto builder=owner.begin(identity,source);assert(builder);
 Owner::Range a,b;Owner::Key ka={17,8},kb={17,9};
 assert(owner.append(builder,ka,source.get(),source->data(),3,projection,-713.25f,208.5f,208.5f,21.18f,a));
 assert(owner.append(builder,ka,source.get(),source->data(),3,projection,-713.25f,208.5f,208.5f,21.18f,b));
 assert(a.first==0 && a.count==3 && b.first==a.first && builder->records==3);
 // Distinct wrapped/native occurrence; canonical shadow borrow stays first.
 assert(owner.append(builder,kb,source.get(),source->data(),3,projection,4810.75f,-263.5f,-263.5f,21.18f,b));
 assert(a.contiguous(b) && builder->records==6);
 auto lease=owner.upload(builder,&device);builder.reset();assert(lease && device.creates==1 && owner.uploads==1);
 assert(owner.cpu_bytes()+owner.gpu_bytes()==owner.bytes() && owner.gpu_bytes()==6*64);
 assert(lease->staging.empty() && lease->find(kb).first==3);
 assert(lease->source(source.get(),21.18f).first==0 && !lease->source(source.get(),21.19f));
 auto data=reinterpret_cast<Owner::Instance const*>(lease->buffer->data.data());
 for(unsigned i=0;i<6;++i){assert(!std::memcmp(data[i].place,(*source)[i%3].place,sizeof(data[i].place)));
  assert(!std::memcmp(data[i].projection,projection,sizeof(projection)) && data[i].view[3]==21.18f);
  assert(data[i].view[0]==(i<3?-713.25f:4810.75f));}
 assert(owner.select(identity)==lease && owner.reuses==1 && device.creates==1);
 auto changed=identity;++changed[0];assert(!owner.select(changed));
 auto next=owner.begin(changed,source);assert(owner.append(next,ka,source.get(),source->data(),1,projection,0,0,0,22,a));
 auto replacement=owner.upload(next,&device);next.reset();assert(replacement && owner.gpu_bytes()==7*64);
 std::weak_ptr<std::vector<Owner::Instance>> retained=source;source.reset();
 owner.clear();replacement.reset();assert(!retained.expired() && device.freed==1 && owner.gpu_bytes()==6*64);
 assert(data[0].view[0]==-713.25f);lease.reset();assert(retained.expired() && device.freed==2 && !owner.bytes());
}
''')

    def test_failure_capacity_transaction_and_joint_pinned_budget(self):
        run_cpp(GPU_STUB + r'''
int main(){
 ID3D11Device device;Owner owner;Owner::Instance instance;float projection[]={0,0,128,1260};
 Owner::Key identity={1},key={1};Owner::Range range;
 auto first=owner.begin(identity);assert(owner.append(first,key,&instance,&instance,1,projection,0,0,0,40,range));
 auto complete=owner.upload(first,&device);first.reset();assert(complete);
 ++identity[0];auto failed=owner.begin(identity);
 assert(!owner.append(failed,key,&instance,&instance,Owner::record_limit+1,projection,0,0,0,40,range));
 assert(failed->records==0 && failed->ranges.empty());
 assert(owner.append(failed,key,&instance,&instance,1,projection,0,0,0,40,range));
 device.fail=true;assert(!owner.upload(failed,&device));assert(owner.select(complete->identity)==complete);
 assert(!failed->complete && !failed->buffer && failed->gpu_bytes()==0);
 device.fail=false;auto recovered=owner.upload(failed,&device);assert(recovered);failed.reset();
 complete.reset();recovered.reset();owner.clear();assert(!owner.bytes());
 std::vector<Owner::Instance> dense(Owner::record_limit);std::vector<Owner::Lease> pinned;
 bool pressure=false;
 for(unsigned n=0;n<12;++n){identity[0]=n+100;auto builder=owner.begin(identity);assert(builder);
  if(!owner.append(builder,key,dense.data(),dense.data(),unsigned(dense.size()),projection,0,0,0,40,range)){pressure=true;break;}
  auto result=owner.upload(builder,&device);builder.reset();
  if(!result){pressure=true;break;}pinned.push_back(result);
  assert(owner.bytes()<=Owner::budget && owner.cpu_bytes()+owner.gpu_bytes()==owner.bytes());
 }
 assert(pressure && pinned.size()>=4 && owner.peak_bytes()<=Owner::budget);
 auto creates=device.creates;assert(owner.select(pinned.back()->identity)==pinned.back() && creates==device.creates);
 owner.clear();assert(owner.bytes()!=0);pinned.clear();assert(owner.bytes()==0 && device.freed==device.creates-1);
 // A subsequent demand recovers after pins retire, with no larger allowance.
 auto again=owner.begin(identity);assert(owner.append(again,key,dense.data(),dense.data(),unsigned(dense.size()),projection,0,0,0,40,range));
 auto result=owner.upload(again,&device);assert(result);again.reset();owner.clear();result.reset();assert(!owner.bytes());
}
''')

    def test_exact_original_and_resident_pass_operation_order(self):
        # Execute production resident entry wrappers with tiny vector/input
        # stubs. The original shader projection functions are consumers here;
        # the assertions cover exact wrappers, fractional/negative anchors,
        # reflection, wrap offsets, and preserved source/material fields.
        import re
        rigid = (ROOT/'Renderer/native/render_core/rigid_feature.hlsl').read_text()
        natural = (ROOT/'Renderer/lab/shared/shaders/objects/instance_geometry.hlsl').read_text()
        caster = (ROOT/'Renderer/native/render_core/rigid_caster.hlsl').read_text()
        wrappers = []
        for text, name in [(rigid,'VSResidentSharedFeature'),(rigid,'VSResidentSharedFeatureReflection'),
                           (caster,'VSResidentSharedCaster')]:
            code = re.search(r'\w+ '+name+r'\([^)]*\)\{.*?\n\}',text,re.S).group()
            wrappers.append(code)
        natural_wrappers = re.findall(r'(?:P|InstancePixel) VSResidentInstance\([^)]*\)\{.*?\n\}',natural,re.S)
        self.assertEqual(len(natural_wrappers),2)
        wrappers += [natural_wrappers[0].replace('VSResidentInstance','VSResidentNaturalCaster'),
                     natural_wrappers[1].replace('VSResidentInstance','VSResidentNaturalBody')]
        run_cpp(r'''
#include <cassert>
#include <cmath>
#include <cstring>
#include <initializer_list>
struct V2 {float x,y;V2 operator+(V2 b)const{return {x+b.x,y+b.y};}};
struct V3 {float x,y,z;};
struct V4 {V2 xy;float z,w;};
struct RigidInput {V4 placement_view;float source_position=0,source_normal=0,source_uv=0;V4 place0{},place1{},projection{};};using InstanceInput=RigidInput;
struct ResidentInstanceInput {float source_position=0,source_normal=0,source_uv=0;unsigned selection=0;};
struct ResidentPlacement {V4 place0{},place1{},projection{},placement_view{};};ResidentPlacement C3XResidentPlacements[1];
RigidInput resident_rigid_input(ResidentInstanceInput input){RigidInput i;i.placement_view=C3XResidentPlacements[input.selection].placement_view;i.source_position=input.source_position;return i;}
InstanceInput resident_natural_input(ResidentInstanceInput input){return resident_rigid_input(input);}
struct FeaturePixelInput {V4 value;float source;};using RigidPixel=FeaturePixelInput;using InstancePixel=FeaturePixelInput;using P=FeaturePixelInput;
V2 c3x_viewport_translation{},translation{};float c3x_viewport_depth_translation=0,depth_translation=0;
struct Offset {V3 xyz;};Offset offset;
FeaturePixelInput VSSharedFeature(RigidInput i){return {i.placement_view,i.source_position};}
FeaturePixelInput VSSharedFeatureReflection(RigidInput i){return {i.placement_view,i.source_position};}
FeaturePixelInput VSSharedCaster(RigidInput i){return {i.placement_view,i.source_position};}
FeaturePixelInput VSInstance(RigidInput i){return {i.placement_view,i.source_position};}
''' + '\n'.join(wrappers).replace('i.placement_view.xyz=offset.xyz',
        'i.placement_view.xy={offset.xyz.x,offset.xyz.y};i.placement_view.z=offset.xyz.z') + r'''
int main(){
 for(float x:{-7312.125f,-.00001f,0.f,5839.75f})for(float y:{-826.875f,.00001f,471.25f})
 for(float camera:{-137.5f,0.f,201.125f}){
  RigidInput input{{{x,y},y,21.18f},71.25f},original=input;
  C3XResidentPlacements[0].placement_view=input.placement_view;ResidentInstanceInput selected{71.25f,0,0,0};
  c3x_viewport_translation=translation={camera,camera*.25f};
  c3x_viewport_depth_translation=depth_translation=camera*1.125f;
  original.placement_view.xy=translation+input.placement_view.xy;
  original.placement_view.z=depth_translation+input.placement_view.z;
  auto expected=VSSharedFeature(original),body=VSResidentSharedFeature(selected),reflected=VSResidentSharedFeatureReflection(selected),tree=VSResidentNaturalBody(selected);
  assert(!std::memcmp(&expected,&body,sizeof(expected)) && !std::memcmp(&expected,&reflected,sizeof(expected)) && !std::memcmp(&expected,&tree,sizeof(expected)));
  offset.xyz={camera,-camera,7.125f};original=input;original.placement_view.xy={camera,-camera};original.placement_view.z=7.125f;
  expected=VSSharedCaster(original);auto shadow=VSResidentSharedCaster(selected),natural_shadow=VSResidentNaturalCaster(selected);
  assert(!std::memcmp(&expected,&shadow,sizeof(expected)) && !std::memcmp(&expected,&natural_shadow,sizeof(expected)));
 }
}
''')


if __name__=='__main__':
    unittest.main()
