#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
#include "Renderer/native/city_fidelity/scene_lights.h"
#include <fstream>
#include <random>
#include <iterator>
int main(){
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
    using namespace c3x_renderer::city_fidelity;
    std::ifstream file("CITY_SHADER_PATH");assert(file);
    std::string shader((std::istreambuf_iterator<char>(file)),{});
    shader+=R"(
StructuredBuffer<float4> Receivers:register(t0);
RWStructuredBuffer<float4> Results:register(u0);
[numthreads(64,1,1)]void CSTest(uint3 id:SV_DispatchThreadID){
 Results[id.x]=float4(q8_local_irradiance(Receivers[id.x*2],Receivers[id.x*2+1].xyz,1),1);
})";
    ComPtr<ID3DBlob> blob,error;
    HRESULT hr=D3DCompile(shader.data(),shader.size(),nullptr,nullptr,nullptr,"CSTest","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&error);
    if(FAILED(hr) && error)std::puts(static_cast<char const*>(error->GetBufferPointer()));checked(hr);
    ComPtr<ID3D11ComputeShader> cs;checked(device->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs));
    Lighting city;unsigned nl=144,nb=96;city.lights.resize(nl);city.blockers.resize(nb);
    std::mt19937 rng(832415);std::uniform_real_distribution<float> u(-1,1);
    for(unsigned i=0;i<nl;++i){auto&l=city.lights[i];float cluster=i%3==0?-3.f:i%3==1?0.f:4.f;
        l.position[0]=cluster+u(rng);l.position[1]=cluster+u(rng);l.position[2]=u(rng)*.5f;
        l.range=.1f+std::abs(u(rng))*1.2f;l.intensity=.8f;l.owner=float(i%nb);
        for(unsigned a=0;a<3;++a){l.direction[a]=u(rng);l.color[a]=.2f+std::abs(u(rng));}}
    for(unsigned j=0;j<nb;++j){float cluster=j%3==0?-3.f:j%3==1?0.f:4.f;
        for(unsigned a=0;a<3;++a){float p=(a==2?0.f:cluster)+u(rng);city.blockers[j].low[a]=p-.1f;city.blockers[j].high[a]=p+.1f;}}
    constexpr unsigned probes=8192;std::vector<std::array<float,4>> receivers(probes*2);
    for(unsigned k=0;k<probes;++k){auto const&p=city.lights[k%nl];
        float x=p.position[0]+u(rng)*p.range,y=p.position[1]+u(rng)*p.range,z=p.position[2]+u(rng)*p.range;
        if(k%5==0){x=std::floor(x*4)*.25f;y=std::floor(y*4)*.25f;}
        // Includes near-horizontal rays and range boundaries at negative coords.
        if(k%7==0){x=std::nextafter(p.position[0]+p.range,p.position[0]);y=p.position[1];z=p.position[2];}
        receivers[k*2]={x,-y,z/source_z_metric,1};receivers[k*2+1]={u(rng),u(rng),u(rng),0};}
    D3D11_BUFFER_DESC d={};d.ByteWidth=unsigned(receivers.size()*16);d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=16;
    D3D11_SUBRESOURCE_DATA init={receivers.data(),0,0};ComPtr<ID3D11Buffer> inputs;checked(device->CreateBuffer(&d,&init,&inputs));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv={};srv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;srv.Buffer.NumElements=probes*2;
    ComPtr<ID3D11ShaderResourceView> input_view;checked(device->CreateShaderResourceView(inputs.Get(),&srv,&input_view));
    d.ByteWidth=probes*16;d.BindFlags=D3D11_BIND_UNORDERED_ACCESS;ComPtr<ID3D11Buffer> output;checked(device->CreateBuffer(&d,nullptr,&output));
    ComPtr<ID3D11UnorderedAccessView> output_view;checked(device->CreateUnorderedAccessView(output.Get(),nullptr,&output_view));
    d.BindFlags=0;d.MiscFlags=0;d.StructureByteStride=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> read;checked(device->CreateBuffer(&d,nullptr,&read));
    SceneLights accelerated,reference;accelerated.options_read=reference.options_read=true;reference.full_scan=true;
    auto run=[&](SceneLights&lights,float night){
        assert(lights.upload(context.Get(),{&city},night,1.1f));
        auto*cb=lights.frame.Get();auto*view=lights.view.Get();auto*input=input_view.Get();auto*out=output_view.Get();
        context->CSSetShader(cs.Get(),nullptr,0);context->CSSetConstantBuffers(6,1,&cb);
        context->CSSetShaderResources(127,1,&view);context->CSSetShaderResources(0,1,&input);context->CSSetUnorderedAccessViews(0,1,&out,nullptr);
        context->Dispatch(probes/64,1,1);out=nullptr;context->CSSetUnorderedAccessViews(0,1,&out,nullptr);
        context->CopyResource(read.Get(),output.Get());D3D11_MAPPED_SUBRESOURCE mapped={};checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&mapped));
        std::vector<float> values(probes*4);std::memcpy(values.data(),mapped.pData,probes*16);context->Unmap(read.Get(),0);return values;
    };
    double max_error=0;unsigned nonzero=0;
    for(float night:{1.f,.35f,0.f}){
        auto expected=run(reference,night),actual=run(accelerated,night);
        for(unsigned k=0;k<actual.size();++k){max_error=std::max(max_error,double(std::abs(expected[k]-actual[k])));
            assert(expected[k]==actual[k]);if(k%4<3 && expected[k]>0)++nonzero;}
    }
    // Content changes at the same pointer, then empty field and a new lifetime.
    city.lights[0].position[0]+=2;assert(run(reference,1)==run(accelerated,1));
    city={};assert(run(reference,1)==run(accelerated,1));
    city.lights.resize(1);city.lights[0].range=1;city.lights[0].intensity=1;city.lights[0].owner=-1;
    city.lights[0].color[0]=1;city.lights[0].direction[0]=1;
    assert(run(reference,1)==run(accelerated,1));
    assert(nonzero>0);std::printf("PASS D3D city lights: 8192 receivers x noon/dusk/night, exact full-scan parity, %u lit channels, max_error=%.9f; mutation/empty/lifetime\n",nonzero,max_error);
}
