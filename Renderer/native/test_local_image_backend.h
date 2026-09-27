#pragma once
#include "gpu_image_compositor.h"
#include "native_hit_scene.h"
namespace c3x_gpu_images {
// Local oracle uses the same ownership adapter as the worker path.
class LocalBackend {
    ID3D11Device* device;ID3D11DeviceContext* context;Compositor gpu;
    c3x_native_hit::Scene hit;
public:
    LocalBackend(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),gpu(d,c){}
    Id create(unsigned w,unsigned h,Format f){auto id=gpu.create(w,h,f);if(id)hit.create(id,w,h,f);return id;}
    bool destroy(Id id){hit.destroy(id);return gpu.destroy(id);}
    bool upload(Id id,std::uint64_t rev,std::uint32_t const* p,std::size_t n){auto ok=gpu.upload(id,rev,p,n);if(ok)hit.upload(id,p,n);return ok;}
    bool submit(Command const* p,std::size_t n){auto ok=gpu.submit(p,n);if(ok)for(std::size_t i=0;i<n;++i)hit.submit(p[i]);return ok;}
    void flush(){}
    bool readback(Id id,std::uint32_t* pixels,std::size_t count){
        auto texture=gpu.texture(id);if(!texture)return false;
        D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);if(count!=std::size_t(d.Width)*d.Height)return false;
        d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
        D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));
        for(unsigned y=0;y<d.Height;++y)std::memcpy(pixels+y*d.Width,static_cast<char*>(m.pData)+y*m.RowPitch,d.Width*4);
        context->Unmap(stage.Get(),0);
        for(unsigned y=0;y<d.Height;++y)for(unsigned x=0;x<d.Width;++x){unsigned value=0;
            if(hit.pixel(id,int(x),int(y),value)&&value!=c3x_native_hit::opaque_map&&value!=pixels[y*d.Width+x])
                throw std::runtime_error("native input coverage disagrees with GPU UI word at "+std::to_string(x)+","+std::to_string(y)+": "+std::to_string(value)+" != "+std::to_string(pixels[y*d.Width+x]));}
        return true;
    }
    Counts stats()const{return gpu.stats();}
};
}
