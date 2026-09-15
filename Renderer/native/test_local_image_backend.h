#pragma once
#include "gpu_image_compositor.h"
namespace c3x_gpu_images {
// Local oracle uses the same ownership adapter as the worker path.
class LocalBackend {
    ID3D11Device* device;ID3D11DeviceContext* context;Compositor gpu;
public:
    LocalBackend(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),gpu(d,c){}
    Id create(unsigned w,unsigned h,Format f){return gpu.create(w,h,f);}
    bool destroy(Id id){return gpu.destroy(id);}
    bool upload(Id id,std::uint64_t rev,std::uint32_t const* p,std::size_t n){return gpu.upload(id,rev,p,n);}
    bool submit(Command const* p,std::size_t n){return gpu.submit(p,n);}
    void flush(){}
    bool readback(Id id,std::uint32_t* pixels,std::size_t count){
        auto texture=gpu.texture(id);if(!texture)return false;
        D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);if(count!=std::size_t(d.Width)*d.Height)return false;
        d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
        D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));
        for(unsigned y=0;y<d.Height;++y)std::memcpy(pixels+y*d.Width,static_cast<char*>(m.pData)+y*m.RowPitch,d.Width*4);
        context->Unmap(stage.Get(),0);return true;
    }
    Counts stats()const{return gpu.stats();}
};
}
