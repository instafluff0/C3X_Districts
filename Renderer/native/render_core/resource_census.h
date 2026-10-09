#pragma once
#include <d3d11.h>
#include <algorithm>
#include <cstddef>
#include <unordered_set>

namespace c3x_renderer { namespace render_core {
// Allocation sizes of live D3D11 resources, computed from their descriptions,
// for the memory census (performance goals, G1). Each resource counts once,
// so overlapping owners cannot double-count a shared texture.
struct ResourceCensus {
    std::unordered_set<ID3D11Resource const*> seen;
    // Bytes per 4x4 block for block-compressed formats, otherwise 0.
    static unsigned block_bytes(DXGI_FORMAT format){
        unsigned f=unsigned(format);
        if((f>=70&&f<=72)||(f>=79&&f<=81))return 8;   // BC1, BC4
        if((f>=73&&f<=78)||(f>=82&&f<=84)||(f>=94&&f<=99))return 16; // BC2, BC3, BC5, BC6H, BC7
        return 0;
    }
    static unsigned pixel_bits(DXGI_FORMAT format){
        unsigned f=unsigned(format);
        if(f>=1&&f<=4)return 128;
        if(f>=5&&f<=8)return 96;
        if(f>=9&&f<=22)return 64;
        if((f>=23&&f<=47)||f==67||(f>=87&&f<=93))return 32;
        if((f>=48&&f<=59)||f==68||f==69||f==85||f==86||f==115)return 16;
        if(f>=60&&f<=65)return 8;
        if(f==66)return 1;
        return 32;
    }
    static std::size_t surface(DXGI_FORMAT format,unsigned width,unsigned height,unsigned mips){
        std::size_t total=0;unsigned block=block_bytes(format),bits=pixel_bits(format);
        for(unsigned level=0;level<(std::max)(1u,mips);++level){
            std::size_t w=(std::max)(1u,width>>level),h=(std::max)(1u,height>>level);
            total+=block?((w+3)/4)*((h+3)/4)*block:(w*h*bits+7)/8;
        }
        return total;
    }
    static std::size_t bytes(ID3D11Resource* resource){
        D3D11_RESOURCE_DIMENSION dimension=D3D11_RESOURCE_DIMENSION_UNKNOWN;resource->GetType(&dimension);
        if(dimension==D3D11_RESOURCE_DIMENSION_BUFFER){
            D3D11_BUFFER_DESC d{};static_cast<ID3D11Buffer*>(resource)->GetDesc(&d);return d.ByteWidth;}
        if(dimension==D3D11_RESOURCE_DIMENSION_TEXTURE1D){
            D3D11_TEXTURE1D_DESC d{};static_cast<ID3D11Texture1D*>(resource)->GetDesc(&d);
            return surface(d.Format,d.Width,1,d.MipLevels)*d.ArraySize;}
        if(dimension==D3D11_RESOURCE_DIMENSION_TEXTURE2D){
            D3D11_TEXTURE2D_DESC d{};static_cast<ID3D11Texture2D*>(resource)->GetDesc(&d);
            return surface(d.Format,d.Width,d.Height,d.MipLevels)*d.ArraySize*(std::max)(1u,d.SampleDesc.Count);}
        if(dimension==D3D11_RESOURCE_DIMENSION_TEXTURE3D){
            D3D11_TEXTURE3D_DESC d{};static_cast<ID3D11Texture3D*>(resource)->GetDesc(&d);
            std::size_t total=0;
            for(unsigned level=0;level<(std::max)(1u,d.MipLevels);++level)
                total+=surface(d.Format,(std::max)(1u,d.Width>>level),(std::max)(1u,d.Height>>level),1)*(std::max)(1u,d.Depth>>level);
            return total;}
        return 0;
    }
    std::size_t add(ID3D11Resource* resource){
        if(!resource||!seen.insert(resource).second)return 0;
        return bytes(resource);
    }
    std::size_t add(ID3D11View* view){
        if(!view)return 0;
        ID3D11Resource* resource=nullptr;view->GetResource(&resource);if(!resource)return 0;
        auto result=add(resource);resource->Release(); // the view keeps it alive
        return result;
    }
};
}}
