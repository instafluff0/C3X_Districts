#pragma once
// One bounded candidate. The existing D24S8 target owns the transient mask;
// these device-local states add no geometry, target, cache or scene owner.
#include <d3d11.h>
#include <wrl/client.h>
struct SandboxUnderlayOcclusion {
    Microsoft::WRL::ComPtr<ID3D11BlendState> no_color;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mark,uncovered;
    ID3D11Device* owner=nullptr;
    bool checked=false;
    bool ensure(ID3D11Device* device){
        if(owner!=device){no_color.Reset();mark.Reset();uncovered.Reset();owner=device;checked=false;}
        if(checked)return no_color && mark && uncovered;
        checked=true;
        D3D11_BLEND_DESC blend={};
        if(FAILED(device->CreateBlendState(&blend,&no_color)))return false;
        D3D11_DEPTH_STENCIL_DESC depth={};
        depth.DepthEnable=TRUE;depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
        depth.StencilEnable=TRUE;depth.StencilReadMask=1;depth.StencilWriteMask=1;
        depth.FrontFace.StencilFunc=D3D11_COMPARISON_ALWAYS;
        depth.FrontFace.StencilFailOp=D3D11_STENCIL_OP_KEEP;
        depth.FrontFace.StencilDepthFailOp=D3D11_STENCIL_OP_KEEP;
        depth.FrontFace.StencilPassOp=D3D11_STENCIL_OP_REPLACE;
        depth.BackFace=depth.FrontFace;
        if(FAILED(device->CreateDepthStencilState(&depth,&mark)))return false;
        depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;
        depth.StencilWriteMask=0;
        depth.FrontFace.StencilFunc=D3D11_COMPARISON_EQUAL;
        depth.FrontFace.StencilPassOp=D3D11_STENCIL_OP_KEEP;
        depth.BackFace=depth.FrontFace;
        return SUCCEEDED(device->CreateDepthStencilState(&depth,&uncovered));
    }
};
