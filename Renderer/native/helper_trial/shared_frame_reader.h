#pragma once
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../asset_content_hash.h"

namespace c3x_helper_trial {
// Diagnostic x86 consumer for an x64-composed immutable frame. The keyed
// mutex releases the shared slot before the helper may draw the next frame.
class SharedFrameReader {
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11Device1> device1;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    unsigned width=0,height=0;
public:
    struct Frame {unsigned width=0,height=0;std::array<std::uint32_t,4> hash={};std::vector<unsigned> pixels;};
    Frame read(std::uint64_t imported_handle,unsigned expected_width,unsigned expected_height){
        if(!imported_handle)throw std::runtime_error("missing shared final image");
        HANDLE handle=reinterpret_cast<HANDLE>(std::uintptr_t(imported_handle));
        if(!device){D3D_FEATURE_LEVEL feature={};
            HRESULT hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context);
            if(FAILED(hr)||FAILED(device.As(&device1))){CloseHandle(handle);throw std::runtime_error("x86 final-image D3D device failed");}}
        Microsoft::WRL::ComPtr<ID3D11Texture2D> source;
        HRESULT hr=device1->OpenSharedResource1(handle,IID_PPV_ARGS(&source));CloseHandle(handle);
        if(FAILED(hr))throw std::runtime_error("x86 final-image import failed");
        D3D11_TEXTURE2D_DESC desc={};source->GetDesc(&desc);
        if(desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM||desc.Width!=expected_width||desc.Height!=expected_height)
            throw std::runtime_error("x86 final-image format or extent differs");
        if(!staging||width!=desc.Width||height!=desc.Height){
            desc.BindFlags=0;desc.MiscFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            staging.Reset();hr=device->CreateTexture2D(&desc,nullptr,&staging);
            if(FAILED(hr))throw std::runtime_error("x86 final-image readback allocation failed");
            width=desc.Width;height=desc.Height;
        }
        Microsoft::WRL::ComPtr<IDXGIKeyedMutex> mutex;
        if(FAILED(source.As(&mutex))||FAILED(mutex->AcquireSync(1,1000)))
            throw std::runtime_error("x86 final-image acquire failed");
        bool mapped=false;
        try{
            context->CopyResource(staging.Get(),source.Get());
            D3D11_MAPPED_SUBRESOURCE pixels={};hr=context->Map(staging.Get(),0,D3D11_MAP_READ,0,&pixels);
            if(FAILED(hr))throw std::runtime_error("x86 final-image readback failed");
            mapped=true;std::vector<unsigned> contiguous(std::size_t(width)*height);
            for(unsigned y=0;y<height;++y)std::memcpy(contiguous.data()+std::size_t(y)*width,
                static_cast<unsigned char const*>(pixels.pData)+std::size_t(y)*pixels.RowPitch,std::size_t(width)*4);
            context->Unmap(staging.Get(),0);mapped=false;
            auto hash=c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(contiguous.data()),contiguous.size()*4);
            if(FAILED(mutex->ReleaseSync(0)))throw std::runtime_error("x86 final-image release failed");
            context->Flush();return {width,height,hash,std::move(contiguous)};
        }catch(...){
            if(mapped)context->Unmap(staging.Get(),0);
            mutex->ReleaseSync(0);context->Flush();throw;
        }
    }
};
}
