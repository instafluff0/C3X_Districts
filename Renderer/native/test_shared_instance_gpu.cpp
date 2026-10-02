// Hardware correctness fixture. Staging readback is test-only, after publication;
// it is neither a production synchronization mechanism nor a performance test.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#pragma comment(lib,"d3d11.lib")
#include "Renderer/native/render_core/shared_instance_submission.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using Owner=c3x_renderer::render_core::SharedInstanceSubmission;
using Instance=Owner::Instance;

namespace {
template<class T> struct Com {
    T* value=nullptr;
    Com()=default;
    Com(Com const&)=delete;
    Com& operator=(Com const&)=delete;
    ~Com(){if(value)value->Release();}
};
void require(bool condition,char const* reason){if(!condition)throw std::runtime_error(reason);}
void checked(HRESULT result,char const* reason){require(!FAILED(result),reason);}
void structured(Owner::Lease const& content){
    require(content && content->buffer && content->view,"missing structured buffer/view");
    D3D11_BUFFER_DESC desc{};content->buffer->GetDesc(&desc);
    require(desc.Usage==D3D11_USAGE_DEFAULT && desc.ByteWidth==content->records*sizeof(Instance) &&
        desc.BindFlags==D3D11_BIND_SHADER_RESOURCE && desc.CPUAccessFlags==0 &&
        desc.MiscFlags==D3D11_RESOURCE_MISC_BUFFER_STRUCTURED && desc.StructureByteStride==sizeof(Instance),
        "production delta buffer descriptor differs");
}
std::vector<unsigned char> readback(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Buffer* source){
    // One staging allocation at a time; the largest fixture buffer is 576 B.
    D3D11_BUFFER_DESC desc{};source->GetDesc(&desc);
    require(desc.ByteWidth && desc.ByteWidth<=9*sizeof(Instance),"readback cap exceeded");
    desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.MiscFlags=0;
    desc.StructureByteStride=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    Com<ID3D11Buffer> staging;
    checked(device->CreateBuffer(&desc,nullptr,&staging.value),"CreateBuffer staging failed");
    std::vector<unsigned char> bytes(desc.ByteWidth);
    context->CopyResource(staging.value,source);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checked(context->Map(staging.value,0,D3D11_MAP_READ,0,&mapped),"test-only staging Map failed");
    std::memcpy(bytes.data(),mapped.pData,bytes.size());context->Unmap(staging.value,0);
    return bytes;
}
void equal(std::vector<unsigned char> const& actual,std::vector<Instance> const& expected,char const* reason){
    require(actual.size()==expected.size()*sizeof(Instance),"unexpected readback byte count");
    require(!std::memcmp(actual.data(),expected.data(),actual.size()),reason);
}
std::uint64_t digest(std::vector<unsigned char> const& bytes){
    std::uint64_t result=14695981039346656037ull;
    for(auto value:bytes){result^=value;result*=1099511628211ull;}return result;
}
std::vector<Instance> placements(unsigned group){
    std::vector<Instance> result(3);
    for(unsigned n=0;n<3;++n){
        for(unsigned k=0;k<8;++k)result[n].place[k]=float(group*100+n*8+k)+.25f;
        for(unsigned k=0;k<4;++k){result[n].projection[k]=-float(k+1);result[n].view[k]=-float(k+5);}
    }
    return result;
}
void expected_append(std::vector<Instance>& expected,std::vector<Instance> const& values,float const* projection,
        float x,float y,float depth,float material){
    for(auto value:values){std::copy(projection,projection+4,value.projection);
        value.view[0]=x;value.view[1]=y;value.view[2]=depth;value.view[3]=material;expected.push_back(value);}
}
Owner::RetainedSource proof(std::shared_ptr<int> const& source,unsigned generation){
    Owner::RetainedSource result;result.source=source;result.owner={generation,generation};
    result.canonical_source=Owner::Generation::source_key(source.get(),40.f);result.canonical=true;return result;
}
void empty_staging(Owner::Lease const& value){
    require(value->placements.empty() && value->staging.empty() && value->copies.empty() && value->updates.empty() &&
        !value->copy_source && !value->content,"completed delta retains CPU placement or old source owner");
}
void test(ID3D11Device* device,ID3D11DeviceContext* context){
    Owner owner;Owner::Range range;
    float const projection[]={17.f,23.f,128.f,1260.f};
    auto a=std::make_shared<int>(11),b=std::make_shared<int>(12),c=std::make_shared<int>(13);
    std::weak_ptr<int> weak_a=a,weak_b=b,weak_c=c;
    auto pa=proof(a,11),pb=proof(b,12),pc=proof(c,13);
    auto va=placements(1),vb=placements(2),vc=placements(3),changed=placements(9);
    std::vector<Instance> old_expected;
    auto builder=owner.begin_retained(Owner::Key{1},{},true);require(bool(builder),"begin first delta failed");
    auto append=[&](Owner::Key const& key,std::shared_ptr<int> const& source,std::vector<Instance> const& values,
            Owner::RetainedSource const& source_proof,float x,float y,float depth){
        require(owner.append(builder,key,source.get(),values.data(),unsigned(values.size()),projection,x,y,depth,40.f,range),
            "append changed range failed");
        require(owner.retain_source(builder,key,source_proof),"retain weak source proof failed");
    };
    append(Owner::Key{11},a,va,pa,0,0,0);expected_append(old_expected,va,projection,0,0,0,40);
    append(Owner::Key{12},b,vb,pb,10,20,20);expected_append(old_expected,vb,projection,10,20,20,40);
    append(Owner::Key{13},c,vc,pc,20,40,40);expected_append(old_expected,vc,projection,20,40,40,40);
    auto old=owner.upload(builder,device,context);builder.reset();structured(old);empty_staging(old);
    require(old->records==9 && owner.uploaded_bytes==9*64 && owner.packed_records==9 && owner.gpu_copies==0,
        "initial packing/upload counters differ");
    std::weak_ptr<Owner::Generation const> weak_old=old;
    unsigned const indices[]={8,0,4};
    auto pinned=owner.prepare_selection(device,old,indices,3,64);
    require(bool(pinned) && pinned->content==old && owner.valid(pinned),"old indexed plan failed");
    require(owner.plan_uploads==1 && owner.plan_uploaded_bytes==sizeof(indices),"indexed plan counters differ");

    // Place C first, a changed B second, then carry A into a new final offset.
    builder=owner.begin_retained(Owner::Key{2},{},true);require(bool(builder),"begin mixed delta failed");
    require(owner.reuse_range(builder,Owner::Key{13},3,pc,range) && range.first==0,"required C reuse failed");
    auto recycled=pc;++recycled.owner[1];
    require(!owner.reuse_range(builder,Owner::Key{13},3,recycled,range),"recycled owner was admitted");
    require(!owner.reuse_range(builder,Owner::Key{12},2,pb,range),"wrong-length source was admitted");
    append(Owner::Key{22},b,changed,pb,9,8,7);
    require(owner.carry_forward(builder,[](auto const& source){return source.owner[1]==11;}),"carry A failed");
    require(builder->records==9 && builder->staging.size()==3 && builder->copies.size()==2 && builder->updates.size()==1 &&
        builder->copy_source==old,"mixed delta repacked retained records or lost copy owner");
    require(owner.packed_records==12 && owner.range_reuses==1 && owner.carried_ranges==1,"host packing reuse counters differ");
    auto before_upload=owner.uploaded_bytes,before_copy=owner.copied_bytes;
    auto next=owner.upload(builder,device,context);builder.reset();structured(next);empty_staging(next);
    require(next->buffer!=old->buffer && owner.uploaded_bytes-before_upload==3*64 && owner.copied_bytes-before_copy==6*64 &&
        owner.gpu_copies==2 && owner.allocated_bytes==18*64,"mixed delta GPU-copy/upload counters differ");
    require(owner.gpu_bytes()==18*64+sizeof(indices),"old front, new front and selected plan not jointly charged");
    auto ca=next->find(Owner::Key{11}),cb=next->find(Owner::Key{22}),cc=next->find(Owner::Key{13});
    require(ca.first==6 && ca.count==3 && cb.first==3 && cb.count==3 && cc.first==0 && cc.count==3,
        "required/carried range offsets differ");
    require(!next->find(Owner::Key{12}) && next->source(a.get(),40,3).first==6 &&
        next->source(b.get(),40,3).first==3 && next->source(c.get(),40,3).first==0,"canonical source mapping differs");
    std::vector<Instance> next_expected;
    next_expected.insert(next_expected.end(),old_expected.begin()+6,old_expected.end());
    expected_append(next_expected,changed,projection,9,8,7,40);
    next_expected.insert(next_expected.end(),old_expected.begin(),old_expected.begin()+3);
    auto old_bytes=readback(device,context,old->buffer),next_bytes=readback(device,context,next->buffer);
    equal(old_bytes,old_expected,"pinned old DEFAULT buffer was mutated");
    equal(next_bytes,next_expected,"mixed DEFAULT replacement bytes differ");
    require(owner.valid(pinned) && pinned->content==old,"replacement invalidated pinned old consumer");
    auto index_bytes=readback(device,context,pinned->buffer);
    require(index_bytes.size()==sizeof(indices) && !std::memcmp(index_bytes.data(),indices,sizeof(indices)),
        "pinned old selected order differs");

    // A GPU-only generation can supply the next replacement without CPU copies.
    std::weak_ptr<Owner::Generation const> weak_next=next;
    builder=owner.begin_retained(Owner::Key{3},{},true);require(bool(builder),"begin GPU-only delta failed");
    require(owner.reuse_range(builder,Owner::Key{22},3,pb,range) && builder->staging.empty(),"GPU-only B reuse failed");
    auto packed=owner.packed_records,uploaded=owner.uploaded_bytes,copied=owner.copied_bytes;
    require(!owner.upload(builder,device,nullptr) && owner.select(Owner::Key{2})==next,
        "invalid context replaced the published front");
    require(owner.packed_records==packed && owner.uploaded_bytes==uploaded && owner.copied_bytes==copied,
        "rejected upload changed counters");
    auto third=owner.upload(builder,device,context);builder.reset();structured(third);empty_staging(third);
    require(owner.packed_records==12 && owner.uploaded_bytes==12*64 && owner.copied_bytes==9*64 &&
        owner.gpu_copies==3 && owner.uploads==3 && owner.allocated_bytes==21*64,"GPU-only replacement counters differ");
    require(owner.gpu_bytes()==21*64+sizeof(indices),"all three pinned GPU generations were not charged");
    next.reset();require(weak_next.expired(),"completed replacement pins previous source generation");
    old.reset();require(!weak_old.expired(),"old indexed consumer did not pin its front");
    a.reset();b.reset();c.reset();
    require(weak_a.expired() && weak_b.expired() && weak_c.expired(),"prepared union holds old mesh/source owners");
    // The upload queued its copy before releasing next; this readback is the first
    // wait for that copy. D3D must retain its referenced source resource itself.
    std::vector<Instance> third_expected;expected_append(third_expected,changed,projection,9,8,7,40);
    auto third_bytes=readback(device,context,third->buffer);
    equal(third_bytes,third_expected,"copy lost source after CPU owner retirement");
    equal(readback(device,context,pinned->content->buffer),old_expected,"old front changed during second replacement");
    pinned.reset();require(weak_old.expired(),"released old consumer still pins its front");
    require(owner.bytes()==owner.cpu_bytes()+owner.gpu_bytes() && owner.gpu_bytes()==3*64 &&
        owner.bytes()<=Owner::budget && owner.peak_bytes()<=Owner::budget,"resident/pinned budget accounting differs");
    auto peak=owner.peak_bytes();
    require(peak>=21*64,"pinned/replacement peak charge was omitted");
    std::printf("{\"event\":\"shared-instance-delta\",\"packed_records\":%llu,\"uploaded_bytes\":%llu,"
        "\"gpu_copies\":%u,\"copied_bytes\":%llu,\"allocated_bytes\":%llu,\"peak_joint_bytes\":%llu,"
        "\"limit_bytes\":%llu,\"old_hash\":\"%016llx\",\"mixed_hash\":\"%016llx\",\"third_hash\":\"%016llx\","
        "\"retired_generation_expired\":true,\"old_consumer_expired\":true,\"weak_mesh_sources_expired\":true}\n",
        static_cast<unsigned long long>(owner.packed_records),static_cast<unsigned long long>(owner.uploaded_bytes),
        owner.gpu_copies,static_cast<unsigned long long>(owner.copied_bytes),static_cast<unsigned long long>(owner.allocated_bytes),
        static_cast<unsigned long long>(peak),static_cast<unsigned long long>(Owner::budget),
        static_cast<unsigned long long>(digest(old_bytes)),static_cast<unsigned long long>(digest(next_bytes)),
        static_cast<unsigned long long>(digest(third_bytes)));
    owner.clear();require(!owner.valid(third) && owner.bytes()>0,"clear failed to retire pinned generation");
    third.reset();require(!owner.bytes() && !owner.cpu_bytes() && !owner.gpu_bytes(),"final owner charges were not released");
}
}

int main(int argc,char** argv){
    try{
        bool warp=false,debug=false;
        for(int n=1;n<argc;++n){if(!std::strcmp(argv[n],"--warp"))warp=true;
            else if(!std::strcmp(argv[n],"--debug"))debug=true;else throw std::runtime_error("unknown fixture option");}
        Com<ID3D11Device> device;Com<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level{};
        D3D_FEATURE_LEVEL const levels[]={D3D_FEATURE_LEVEL_11_0};
        checked(D3D11CreateDevice(nullptr,warp?D3D_DRIVER_TYPE_WARP:D3D_DRIVER_TYPE_HARDWARE,nullptr,
            debug?D3D11_CREATE_DEVICE_DEBUG:0,levels,1,D3D11_SDK_VERSION,&device.value,&level,&context.value),
            "D3D11CreateDevice failed (hardware is required by default)");
        require(level==D3D_FEATURE_LEVEL_11_0,"feature level 11 required");
        std::printf("{\"event\":\"device\",\"backend\":\"%s\",\"feature_level\":%u,\"debug\":%s}\n",
            warp?"warp":"hardware",unsigned(level),debug?"true":"false");
        test(device.value,context.value);
        checked(device.value->GetDeviceRemovedReason(),"D3D device removed during fixture");
        std::puts("{\"event\":\"result\",\"passed\":true,\"performance_measurement\":false}");return 0;
    }catch(std::exception const& error){
        std::fprintf(stderr,"shared-instance-gpu-test FAILED: %s\n",error.what());return 1;
    }
}
