#define NOMINMAX
#include <windows.h>
#include "unit_pose_content.h"
#include "gpu_unit_shadow.h"
#include "gpu_unit_finish.h"
#include <cassert>
#include <cstdio>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_renderer;
using Microsoft::WRL::ComPtr;
int test_gpu_unit_shadow(){
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
    assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)));
    auto mesh=std::make_shared<AnimationMesh>();mesh->bones=1;mesh->frames=2;mesh->duration=1;
    mesh->vertices.resize(6);mesh->indices={0,1,2,3,4,5,2,1,0};
    for(unsigned i=0;i<6;++i){auto& v=mesh->vertices[i];
        v.source.position[0]=i%3==1?.237f:-.217f;v.source.position[1]=i%3==2?.331f:-.193f;
        v.source.position[2]=i<3?.31f:(i%3==0?-.09f:.57f);v.source.normal[2]=1;
        v.tangent={1,0,0};v.bitangent={0,1,0};v.weights={1,0,0,0};}
    for(unsigned f=0;f<2;++f)for(float v:{1.f,0.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,0.f,1.f,0.f,float(f)*.03f,0.f,0.f,1.f})mesh->palettes.push_back(v);
    GpuUnitShadow pass;GpuUnitFinish finish;std::atomic<bool> cancel{false};unsigned checked=0,coverage_errors=0,finish_errors=0;double max_error=0;
    for(int extent:{128,1536})for(int direction:{1,3,6,8})for(float zoom:{.5f,1.25f}){
        UnitPoseInput input;input.source.meshes={mesh};input.source.shadow_extent=extent;input.source.allow_exit_clip=true;
        input.width=127;input.height=109;input.anchor_x=64;input.anchor_y=72;input.zoom=zoom;
        input.direction=direction;input.phase=.375;input.light_x=.46f;input.light_y=-.83f;
        auto cpu=UnitPoseCompiler{}(input,cancel,0);input.gpu_shadow=true;auto gpu=UnitPoseCompiler{}(input,cancel,0);assert(cpu&&gpu);
        D3D11_TEXTURE2D_DESC desc={};desc.Width=desc.Height=extent;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_R32_FLOAT;desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        ComPtr<ID3D11Texture2D> heights,read;ComPtr<ID3D11RenderTargetView> target;ComPtr<ID3D11ShaderResourceView> view;
        assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&heights))&&SUCCEEDED(device->CreateRenderTargetView(heights.Get(),nullptr,&target))&&SUCCEEDED(device->CreateShaderResourceView(heights.Get(),nullptr,&view)));
        pass.draw(device.Get(),context.Get(),target.Get(),extent,gpu->shadow_triangles);
        desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&read)));context->CopyResource(read.Get(),heights.Get());
        D3D11_MAPPED_SUBRESOURCE map={};assert(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&map)));
        for(int y=0;y<extent;++y)for(int x=0;x<extent;++x){float actual=reinterpret_cast<float*>(static_cast<char*>(map.pData)+y*map.RowPitch)[x],expected=cpu->shadow.heights[y*extent+x];
            if((actual<0)!=(expected<0)){if(coverage_errors++<12)std::printf("SHADOW_DIFF extent=%d direction=%d zoom=%g xy=%d,%d actual=%.9g expected=%.9g\n",extent,direction,zoom,x,y,actual,expected);}
            else {max_error=std::max(max_error,double(std::abs(actual-expected)));assert(std::abs(actual-expected)<1e-6f);}}
        context->Unmap(read.Get(),0);
        desc={};desc.Width=input.width;desc.Height=input.height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        std::vector<unsigned> pixels(input.width*input.height);for(unsigned i=0;i<pixels.size();++i)pixels[i]=((i%256)<<24)|0x754c21;
        D3D11_SUBRESOURCE_DATA data={pixels.data(),UINT(input.width*4),0};ComPtr<ID3D11Texture2D> body;
        assert(SUCCEEDED(device->CreateTexture2D(&desc,&data,&body)));
        auto result=finish.finish(device.Get(),context.Get(),body.Get(),input.width,input.height,view.Get(),gpu->ground_projection);
        result->GetDesc(&desc);desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;read.Reset();
        assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&read)));context->CopyResource(read.Get(),result.Get());
        assert(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&map)));
        for(int y=0;y<input.height;++y)for(int x=0;x<input.width;++x){auto index=y*input.width+x;unsigned p=pixels[index],alpha=p>>24,shade=cpu->ground_shadow[index];
            unsigned expected=(alpha+(shade*(255-alpha)+127)/255)<<24;
            for(unsigned shift:{0u,8u,16u})expected|=(((((p>>shift)&255)*alpha+127)/255)<<shift);
            unsigned actual=reinterpret_cast<unsigned*>(static_cast<char*>(map.pData)+y*map.RowPitch)[x];if(actual!=expected){if(finish_errors++<12)std::printf("FINISH_DIFF extent=%d direction=%d zoom=%g xy=%d,%d actual=%08x expected=%08x\n",extent,direction,zoom,x,y,actual,expected);}++checked;}
        context->Unmap(read.Get(),0);
    }
    std::printf("coverage_errors=%u finish_errors=%u\n",coverage_errors,finish_errors);std::fflush(stdout);assert(!coverage_errors&&!finish_errors);
    pass.reset();finish.reset();
    std::printf("PASS selected GPU shadow pass: exact coverage and %u finished pixels; maximum height error %.9g; directions/zoom/full 1536 resolution, reset\n",checked,max_error);
    return 0;
}
