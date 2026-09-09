#pragma once
// One guarded off-screen reflection scratch target; no presenter or readback.
namespace c3x_renderer { namespace environment_refresh {
struct Reflection {
    render_core::LinearTarget linear;
    ID3D11VertexShader*vs[5]={};ID3D11PixelShader*ps[5]={};
    ID3D11Buffer*frame=nullptr;
    float height_pixels=0,depth_metric=0;
    bool enabled=true;
    unsigned native_extent=136,native_height=136;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){linear.reset();for(auto&p:vs)drop(p);for(auto&p:ps)drop(p);drop(frame);}
    ~Reflection(){reset();}
    bool ensure(ID3D11Device*device,std::string const&root,char const*directory="environment_refresh",unsigned extent=136,unsigned height=0){
        if(!height)height=extent;
        if(frame && native_extent==extent && native_height==height)return true;
        reset();native_extent=extent;native_height=height;
        char const*names[]={"hydrology","feature","terrain","mountain","objects"};
        for(unsigned i=0;i<5;i++){
            std::string path=root+"/Renderer/native/"+directory+"/"+names[i]+".hlsl";
            std::wstring wide(path.begin(),path.end());ID3DBlob*v=nullptr,*p=nullptr,*errors=nullptr;
            HRESULT hr=render_core::compile_cached(wide.c_str(),"VSReflection","vs_5_0",&v,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
            if(SUCCEEDED(hr))hr=render_core::compile_cached(wide.c_str(),"PSReflection","ps_5_0",&p,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
            if(SUCCEEDED(hr))hr=device->CreateVertexShader(v->GetBufferPointer(),v->GetBufferSize(),nullptr,&vs[i]);
            if(SUCCEEDED(hr))hr=device->CreatePixelShader(p->GetBufferPointer(),p->GetBufferSize(),nullptr,&ps[i]);
            drop(v);drop(p);if(FAILED(hr)){reset();return false;}
        }
        D3D11_BUFFER_DESC d={};d.ByteWidth=32;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(FAILED(device->CreateBuffer(&d,nullptr,&frame))){reset();return false;}return true;
    }
    void bind(ID3D11DeviceContext*context){
        float values[]={height_pixels,depth_metric,2.5f/112.f,enabled?1.f:0.f,float(native_extent*2),float(native_height*2),8,8};
        context->UpdateSubresource(frame,0,nullptr,values,0,0);
        context->VSSetConstantBuffers(5,1,&frame);context->PSSetConstantBuffers(5,1,&frame);
    }
    void resolve(ID3D11DeviceContext*context){
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->ResolveSubresource(linear.resolved,0,linear.color,0,DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
};
} }
