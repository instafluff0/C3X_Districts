#pragma once
namespace c3x_renderer { namespace city_fidelity {
struct Glow {
    ID3D11ComputeShader*shader=nullptr;ID3D11Buffer*settings=nullptr;
    ID3D11Texture2D*color=nullptr,*validity=nullptr,*native=nullptr;
    ID3D11ShaderResourceView*view=nullptr;ID3D11UnorderedAccessView*output=nullptr,*output_validity=nullptr;
    ID3D11RenderTargetView*target=nullptr;
    render_core::LinearTarget linear;
    float gain=6;
    unsigned native_extent=136,native_height=136;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){linear.reset();drop(target);drop(native);drop(output_validity);drop(output);drop(view);drop(validity);drop(color);drop(settings);drop(shader);}
    ~Glow(){reset();}
    bool ensure(ID3D11Device*device,std::string const&root,unsigned extent=136,unsigned height=0){
        if(!height)height=extent;
        if(shader && native_extent==extent && native_height==height)return true;
        if(!((extent==136 || extent==264 || extent==520) && height==extent) && !(extent==2248 && height==264))return false;
        reset();native_extent=extent;native_height=height;
        std::wstring path(root.begin(),root.end());path+=L"/Renderer/native/city_fidelity/hdr_glow.hlsl";
        ID3DBlob*blob=nullptr,*errors=nullptr;
        HRESULT hr=render_core::compile_cached(path.c_str(),"CSPost","cs_5_0",&blob,&errors);
        if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=device->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&shader);drop(blob);
        D3D11_BUFFER_DESC b={};b.ByteWidth=48;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&settings);
        D3D11_TEXTURE2D_DESC d={};d.Width=native_extent;d.Height=native_height;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
        d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        if(SUCCEEDED(hr))hr=device->CreateTexture2D(&d,nullptr,&color);
        if(SUCCEEDED(hr))hr=device->CreateShaderResourceView(color,nullptr,&view);
        if(SUCCEEDED(hr))hr=device->CreateUnorderedAccessView(color,nullptr,&output);
        d.Format=DXGI_FORMAT_R32_FLOAT;d.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
        if(SUCCEEDED(hr))hr=device->CreateTexture2D(&d,nullptr,&validity);
        if(SUCCEEDED(hr))hr=device->CreateUnorderedAccessView(validity,nullptr,&output_validity);
        d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.BindFlags=D3D11_BIND_RENDER_TARGET;
        if(SUCCEEDED(hr))hr=device->CreateTexture2D(&d,nullptr,&native);
        if(SUCCEEDED(hr))hr=device->CreateRenderTargetView(native,nullptr,&target);
        if(FAILED(hr) || !linear.ensure(device,native_extent*2,native_height*2)){reset();return false;}return true;
    }
    static D3D11_RECT dispatch_rectangle(unsigned width,unsigned height,D3D11_RECT const* dirty){
        D3D11_RECT rect={0,0,LONG(width),LONG(height)};
        if(dirty){
            rect.left=std::clamp<LONG>(dirty->left,0,LONG(width));
            rect.top=std::clamp<LONG>(dirty->top,0,LONG(height));
            rect.right=std::clamp<LONG>(dirty->right,rect.left,LONG(width));
            rect.bottom=std::clamp<LONG>(dirty->bottom,rect.top,LONG(height));
        }
        if(rect.left==rect.right || rect.top==rect.bottom)return {0,0,0,0};
        rect.left=rect.left/8*8;rect.top=rect.top/8*8;
        rect.right=(rect.right+7)/8*8;rect.bottom=(rect.bottom+7)/8*8;
        return rect;
    }
    std::size_t reconstruct(ID3D11DeviceContext*context,D3D11_RECT const* dirty=nullptr){
        auto dispatch=dispatch_rectangle(native_extent,native_height,dirty);
        if(dispatch.right==dispatch.left || dispatch.bottom==dispatch.top)return 0;
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->ResolveSubresource(linear.resolved,0,linear.color,0,DXGI_FORMAT_R16G16B16A16_FLOAT);
        int extent=static_cast<int>(native_extent),height=static_cast<int>(native_height);
        struct {unsigned input[2],output_size[2];int rectangle[4];float glow[4];} values={{native_extent*2,native_height*2},{native_extent,native_height},{0,0,extent,height},{gain,float(dispatch.left),float(dispatch.top),0}};
        context->UpdateSubresource(settings,0,nullptr,&values,0,0);
        context->CSSetShader(shader,nullptr,0);context->CSSetConstantBuffers(2,1,&settings);
        context->CSSetShaderResources(0,1,&linear.view);context->CSSetShaderResources(3,1,&linear.view);
        context->CSSetUnorderedAccessViews(1,1,&output,nullptr);context->CSSetUnorderedAccessViews(4,1,&output_validity,nullptr);
        context->Dispatch(UINT((dispatch.right-dispatch.left)/8),UINT((dispatch.bottom-dispatch.top)/8),1);
        ID3D11ShaderResourceView*empty=nullptr;ID3D11UnorderedAccessView*unused=nullptr;
        context->CSSetShaderResources(0,1,&empty);context->CSSetShaderResources(3,1,&empty);
        context->CSSetUnorderedAccessViews(1,1,&unused,nullptr);context->CSSetUnorderedAccessViews(4,1,&unused,nullptr);
        context->CSSetShader(nullptr,nullptr,0);
        return std::size_t(dispatch.right-dispatch.left)*(dispatch.bottom-dispatch.top);
    }
};
} }
