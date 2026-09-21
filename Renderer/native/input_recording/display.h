#pragma once
#include "codec.h"
#include <filesystem>
#include <fstream>
namespace c3x_inputs {
// Replay-only readback of the actual retained presenter surface, including
// native UI partial transfers and ambient-only frames. Never a game input.
inline bool display_pixels(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* source,
                           std::vector<unsigned>& pixels,unsigned& width,unsigned& height){
    if(!device||!context||!source)return false;
    D3D11_TEXTURE2D_DESC desc={};source->GetDesc(&desc);
    require(desc.Width&&desc.Width<=2240&&desc.Height&&desc.Height<=1260&&desc.Format==DXGI_FORMAT_B8G8R8A8_UNORM&&desc.SampleDesc.Count==1,"unsupported replay display");
    width=desc.Width;height=desc.Height;desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;desc.MiscFlags=0;
    ID3D11Texture2D* stage=nullptr;if(FAILED(device->CreateTexture2D(&desc,nullptr,&stage)))return false;
    struct Release {ID3D11Texture2D* value;~Release(){value->Release();}} release{stage};
    pixels.resize(std::size_t(width)*height);context->CopyResource(stage,source);D3D11_MAPPED_SUBRESOURCE mapped={};
    if(FAILED(context->Map(stage,0,D3D11_MAP_READ,0,&mapped)))return false;
    for(unsigned y=0;y<height;++y)std::memcpy(pixels.data()+std::size_t(y)*width,static_cast<unsigned char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,std::size_t(width)*4);
    context->Unmap(stage,0);return true;
}
inline void display_bitmap(wchar_t const* path,std::vector<unsigned> const& pixels,unsigned width,unsigned height){
    require(width&&height&&pixels.size()==std::size_t(width)*height,"invalid replay display pixels");
    Writer header;header.u32(54+unsigned(pixels.size())*4);header.u32(0);header.u32(54);header.u32(40);header.u32(width);header(std::int32_t(-std::int32_t(height)));
    header.u32(0x200001);header.u32(0);header.u32(unsigned(pixels.size())*4);header.u32(0);header.u32(0);header.u32(0);header.u32(0);
    auto file=std::filesystem::path(path);require(!std::filesystem::exists(file),"frame export already exists");std::ofstream output(file,std::ios::binary);
    output.write("BM",2);output.write(reinterpret_cast<char const*>(header.bytes.data()),std::streamsize(header.bytes.size()));
    output.write(reinterpret_cast<char const*>(pixels.data()),std::streamsize(pixels.size()*4));output.flush();require(bool(output),"frame export failed");
}
}
