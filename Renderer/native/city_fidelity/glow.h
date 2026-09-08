#pragma once
namespace c3x_renderer { namespace city_fidelity {
struct Glow {
    ID3D11ComputeShader*shader=nullptr;ID3D11Buffer*settings=nullptr;
    ID3D11Texture2D*color=nullptr,*validity=nullptr,*native=nullptr;
    ID3D11ShaderResourceView*view=nullptr;ID3D11UnorderedAccessView*output=nullptr,*output_validity=nullptr;
    ID3D11RenderTargetView*target=nullptr;
    render_core::LinearTarget linear;
    float gain=6;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){linear.reset();drop(target);drop(native);drop(output_validity);drop(output);drop(view);drop(validity);drop(color);drop(settings);drop(shader);}
    ~Glow(){reset();}
    bool ensure(ID3D11Device*device,std::string const&root){
        if(shader)return true;
        std::wstring path(root.begin(),root.end());path+=L"/Renderer/native/city_fidelity/hdr_glow.hlsl";
        ID3DBlob*blob=nullptr,*errors=nullptr;
        HRESULT hr=render_core::compile_cached(path.c_str(),"CSPost","cs_5_0",&blob,&errors);
        if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=device->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&shader);drop(blob);
        D3D11_BUFFER_DESC b={};b.ByteWidth=48;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&settings);
        D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=136;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
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
        if(FAILED(hr) || !linear.ensure(device,272,272)){reset();return false;}return true;
    }
    void reconstruct(ID3D11DeviceContext*context){
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->ResolveSubresource(linear.resolved,0,linear.color,0,DXGI_FORMAT_R16G16B16A16_FLOAT);
        struct {unsigned input[2],output_size[2];int rectangle[4];float glow[4];} values={{272,272},{136,136},{0,0,136,136},{gain,0,0,0}};
        context->UpdateSubresource(settings,0,nullptr,&values,0,0);
        context->CSSetShader(shader,nullptr,0);context->CSSetConstantBuffers(2,1,&settings);
        context->CSSetShaderResources(0,1,&linear.view);context->CSSetShaderResources(3,1,&linear.view);
        context->CSSetUnorderedAccessViews(1,1,&output,nullptr);context->CSSetUnorderedAccessViews(4,1,&output_validity,nullptr);
        context->Dispatch(17,17,1);
        ID3D11ShaderResourceView*empty=nullptr;ID3D11UnorderedAccessView*unused=nullptr;
        context->CSSetShaderResources(0,1,&empty);context->CSSetShaderResources(3,1,&empty);
        context->CSSetUnorderedAccessViews(1,1,&unused,nullptr);context->CSSetUnorderedAccessViews(4,1,&unused,nullptr);
        context->CSSetShader(nullptr,nullptr,0);
    }
};
} }
