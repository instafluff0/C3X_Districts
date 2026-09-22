#define NOMINMAX
#include <windows.h>
#include "gpu_image_compositor.h"
#include "render_core/linear_target.h"
#include <cstdio>
using namespace c3x_gpu_images;
using namespace c3x_renderer::render_core;
void verify_redraw(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
ComPtr<ID3DBlob> redraw_shader(char const* source,char const* entry,char const* model){
    ComPtr<ID3DBlob> blob,errors;auto hr=D3DCompile(source,std::strlen(source),"redraw_oracle",nullptr,nullptr,entry,model,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errors);
    if(errors)std::fprintf(stderr,"%s",static_cast<char*>(errors->GetBufferPointer()));checked(hr);return blob;
}
int main(){try{
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
    char const* seed=R"(
cbuffer Settings:register(b0){float phase;float3 unused;};
float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
struct Output{float4 color:SV_Target;float depth:SV_Depth;};
Output PS(float4 p:SV_Position,uint sample:SV_SampleIndex){Output o;
 o.color=float4(p.x/32+phase,p.y/32,sample*2.125,1);o.depth=(sample+1)*.12345+p.x*.0000001+phase*.01;return o;}
)";
    char const* compare=R"(
Texture2DMS<float4,4> a:register(t0),b:register(t1);
Texture2DMS<float,4> da:register(t2),db:register(t3);
RWTexture2D<uint> result:register(u0);
[numthreads(8,8,1)]void CS(uint3 p:SV_DispatchThreadID){uint w,h;result.GetDimensions(w,h);if(p.x>=w||p.y>=h)return;
 uint difference=0;for(int s=0;s<4;++s)difference|=any(asuint(a.Load(p.xy,s))!=asuint(b.Load(p.xy,s)))||asuint(da.Load(p.xy,s))!=asuint(db.Load(p.xy,s));result[p.xy]=difference;}
)";
    ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps;ComPtr<ID3D11ComputeShader> cs;
    auto blob=redraw_shader(seed,"VS","vs_5_0");checked(device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vs));
    blob=redraw_shader(seed,"PS","ps_5_0");checked(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&ps));
    blob=redraw_shader(compare,"CS","cs_5_0");checked(device->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs));
    D3D11_BUFFER_DESC bd={};bd.ByteWidth=16;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;ComPtr<ID3D11Buffer> constants;checked(device->CreateBuffer(&bd,nullptr,&constants));
    LinearRestore clear;verify_redraw(clear.ensure(device.Get()),"clear shader");
    unsigned checks=0;
    for(unsigned w:{32u,48u,544u}){
        unsigned h=w>256?280:24;LinearTarget source,reference,actual;
        for(auto target:{&source,&reference,&actual})verify_redraw(target->ensure(device.Get(),w,h,true,false),"oracle target allocation");
        D3D11_TEXTURE2D_DESC td={};td.Width=w;td.Height=h;td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R32_UINT;td.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
        ComPtr<ID3D11Texture2D> differences,readback;ComPtr<ID3D11UnorderedAccessView> output;
        checked(device->CreateTexture2D(&td,nullptr,&differences));checked(device->CreateUnorderedAccessView(differences.Get(),nullptr,&output));
        td.BindFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&td,nullptr,&readback));
        auto fill=[&](LinearTarget& target,float phase,D3D11_RECT const* region=nullptr){
            float values[4]={phase};context->UpdateSubresource(constants.Get(),0,nullptr,values,0,0);auto cb=constants.Get();context->PSSetConstantBuffers(0,1,&cb);
            context->OMSetRenderTargets(1,&target.target,target.depth);context->OMSetDepthStencilState(clear.depth,0);context->OMSetBlendState(nullptr,nullptr,~0u);context->RSSetState(clear.rasterizer);
            D3D11_VIEWPORT vp={0,0,float(w),float(h),0,1};D3D11_RECT scissor={0,0,LONG(w),LONG(h)};context->RSSetViewports(1,&vp);if(region)scissor={region->left*2,region->top*2,region->right*2,region->bottom*2};context->RSSetScissorRects(1,&scissor);
            context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);context->VSSetShader(vs.Get(),nullptr,0);context->PSSetShader(ps.Get(),nullptr,0);context->Draw(3,0);context->OMSetRenderTargets(0,nullptr,nullptr);
        };
        auto equal=[&]{
            ID3D11ShaderResourceView* inputs[]={reference.samples,actual.samples,reference.depth_samples,actual.depth_samples};auto out=output.Get();
            context->CSSetShader(cs.Get(),nullptr,0);context->CSSetShaderResources(0,4,inputs);context->CSSetUnorderedAccessViews(0,1,&out,nullptr);context->Dispatch((w+7)/8,(h+7)/8,1);
            for(auto& v:inputs)v=nullptr;out=nullptr;context->CSSetShaderResources(0,4,inputs);context->CSSetUnorderedAccessViews(0,1,&out,nullptr);context->CSSetShader(nullptr,nullptr,0);
            context->CopyResource(readback.Get(),differences.Get());D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&m));bool same=true;
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)same=same&&!reinterpret_cast<unsigned*>(static_cast<char*>(m.pData)+y*m.RowPitch)[x];context->Unmap(readback.Get(),0);return same;
        };
        std::vector<D3D11_RECT> full={{0,0,LONG(w/2),LONG(h/2)}};
        fill(reference,0);fill(actual,0);
        for(unsigned phase=1;phase<6;++phase){
            std::vector<D3D11_RECT> region={{LONG(phase),1,LONG(w/2-1),LONG(h/2-2)}};
            fill(source,float(phase));
            verify_redraw(clear.draw(context.Get(),reference,source.samples,source.depth_samples,0,0,{},&region),"reference exact sample copy");
            verify_redraw(clear.draw(context.Get(),actual,nullptr,nullptr,0,0,{},&region,0,0,false,true),"partial color/depth clear");
            fill(actual,float(phase),&region[0]);
            verify_redraw(equal(),"geometry redraw differs in HDR color/depth or outside damage");++checks;
            fill(actual,99,&region[0]);verify_redraw(!equal(),"negative control must distinguish changed samples");
            fill(actual,float(phase),&region[0]);verify_redraw(equal(),"negative control recovery");
        }
        verify_redraw(clear.draw(context.Get(),actual,nullptr,nullptr,0,0,{},&full,0,0,false,true),"full clear after removal");
        fill(actual,0);fill(reference,0);verify_redraw(equal(),"removed animation restores static scene");++checks;
    }
    context->ClearState();std::printf("PASS retained geometry redraw: checks=%u every_color_depth_sample_exact=1 partial_redraw_clear=1 resize=1 negative_control=1\n",checks);return 0;
}catch(std::exception const& e){std::fprintf(stderr,"FAIL retained geometry redraw: %s\n",e.what());return 1;}}
