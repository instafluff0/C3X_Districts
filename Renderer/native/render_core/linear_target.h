#pragma once
// Worker-owned D3D11 scratch targets. Pixels enter production caches only after
// the common scene-linear MSAA resolve and display transfer have completed.
namespace c3x_renderer { namespace render_core {
struct LinearTarget {
    ID3D11Texture2D *color=nullptr,*resolved=nullptr,*depth_texture=nullptr;
    ID3D11RenderTargetView *target=nullptr;
    ID3D11ShaderResourceView *view=nullptr;
    ID3D11DepthStencilView *depth=nullptr;
    UINT width=0,height=0;
    template<class T> void release(T*& p) { if(p) { p->Release(); p=nullptr; } }
    void reset() {
        release(depth); release(depth_texture); release(view); release(target);
        release(resolved); release(color); width=height=0;
    }
    ~LinearTarget() { reset(); }
    bool ensure(ID3D11Device* device,UINT w,UINT h) {
        if(target && width==w && height==h) return true;
        reset();
        UINT quality=0;
        if(FAILED(device->CheckMultisampleQualityLevels(DXGI_FORMAT_R16G16B16A16_FLOAT,4,&quality)) ||
            quality==0) return false;
        D3D11_TEXTURE2D_DESC d={}; d.Width=w; d.Height=h; d.MipLevels=d.ArraySize=1;
        d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT; d.SampleDesc.Count=4;
        d.BindFlags=D3D11_BIND_RENDER_TARGET; d.Usage=D3D11_USAGE_DEFAULT;
        HRESULT hr=device->CreateTexture2D(&d,nullptr,&color);
        if(SUCCEEDED(hr)) hr=device->CreateRenderTargetView(color,nullptr,&target);
        d.SampleDesc.Count=1; d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        if(SUCCEEDED(hr)) hr=device->CreateTexture2D(&d,nullptr,&resolved);
        if(SUCCEEDED(hr)) hr=device->CreateShaderResourceView(resolved,nullptr,&view);
        d.SampleDesc.Count=4; d.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
        d.BindFlags=D3D11_BIND_DEPTH_STENCIL;
        if(SUCCEEDED(hr)) hr=device->CreateTexture2D(&d,nullptr,&depth_texture);
        if(SUCCEEDED(hr)) hr=device->CreateDepthStencilView(depth_texture,nullptr,&depth);
        if(FAILED(hr)) { reset(); return false; }
        width=w; height=h; return true;
    }
    std::size_t bytes() const { return std::size_t(width)*height*(8*4+8+4*4); }
};
struct LinearOutput {
    ID3D11VertexShader *vertex=nullptr;
    ID3D11PixelShader *pixel=nullptr;
    ID3D11Buffer *settings=nullptr;
    template<class T> void release(T*& p) { if(p) { p->Release(); p=nullptr; } }
    void reset() { release(settings); release(pixel); release(vertex); }
    ~LinearOutput() { reset(); }
    bool ensure(ID3D11Device* device) {
        if(pixel) return true;
        // Matches the pinned Q6 response after reconstruction. Keep alpha
        // straight for the existing Civ III surface-copy contract.
        char const* source=R"(
Texture2D<float4> scene : register(t0);
cbuffer OutputSettings : register(b0) { float exposure; int render_scale; float2 padding; };
float4 VSOutput(uint id : SV_VertexID) : SV_Position {
 float2 p=float2((id<<1)&2,id&2); return float4(p*float2(2,-2)+float2(-1,1),0,1);
}
float4 PSOutput(float4 position : SV_Position) : SV_Target {
 float4 c=0;int2 start=int2(position.xy)*render_scale;
 for(int y=0;y<render_scale;y++)for(int x=0;x<render_scale;x++)
  c+=scene.Load(int3(start+int2(x,y),0));
 c/=render_scale*render_scale;
 if(c.a<=.000001) return 0;
 float3 rgb=max(0,c.rgb/c.a*exposure);
 rgb/=1+max(rgb.r,max(rgb.g,rgb.b));
 rgb=float3(rgb.r<=.0031308?rgb.r*12.92:1.055*pow(rgb.r,1/2.4)-.055,
            rgb.g<=.0031308?rgb.g*12.92:1.055*pow(rgb.g,1/2.4)-.055,
            rgb.b<=.0031308?rgb.b*12.92:1.055*pow(rgb.b,1/2.4)-.055);
 return float4(saturate(rgb),saturate(c.a));
})";
        auto compile=[&](char const* entry,char const* target,ID3DBlob** blob) {
            ID3DBlob* errors=nullptr;
            HRESULT hr=D3DCompile(source,std::strlen(source),"render_core_output",nullptr,nullptr,
                entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,blob,&errors);
            if(errors) { OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer())); errors->Release(); }
            return SUCCEEDED(hr);
        };
        ID3DBlob *vs=nullptr,*ps=nullptr;
        bool ok=compile("VSOutput","vs_4_0",&vs) && compile("PSOutput","ps_4_0",&ps);
        HRESULT hr=E_FAIL;
        if(ok) hr=device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex);
        if(SUCCEEDED(hr)) hr=device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel);
        release(vs); release(ps);
        D3D11_BUFFER_DESC d={}; d.ByteWidth=16; d.Usage=D3D11_USAGE_DEFAULT;
        d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr)) hr=device->CreateBuffer(&d,nullptr,&settings);
        if(FAILED(hr)) { reset(); return false; } return true;
    }
    void draw(ID3D11DeviceContext* context,LinearTarget const& linear,
              ID3D11RenderTargetView* destination,float exposure,int render_scale=1,
              ID3D11ShaderResourceView* reconstructed=nullptr,UINT reconstructed_width=0,UINT reconstructed_height=0) {
        context->OMSetRenderTargets(0,nullptr,nullptr);
        if(!reconstructed)context->ResolveSubresource(linear.resolved,0,linear.color,0,DXGI_FORMAT_R16G16B16A16_FLOAT);
        context->OMSetRenderTargets(1,&destination,nullptr);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(nullptr,0);
        UINT width=reconstructed?reconstructed_width:linear.width/render_scale,height=reconstructed?reconstructed_height:linear.height/render_scale;
        D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};
        D3D11_RECT rect={0,0,LONG(width),LONG(height)};
        context->RSSetViewports(1,&viewport); context->RSSetScissorRects(1,&rect);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0); context->PSSetShader(pixel,nullptr,0);
        struct {float exposure;int scale;float padding[2];}values={exposure,render_scale,{0,0}};
        context->UpdateSubresource(settings,0,nullptr,&values,0,0);
        context->PSSetConstantBuffers(0,1,&settings);
        ID3D11ShaderResourceView*source=reconstructed?reconstructed:linear.view;
        context->PSSetShaderResources(0,1,&source);
        context->Draw(3,0);
        ID3D11ShaderResourceView* empty=nullptr;
        context->PSSetShaderResources(0,1,&empty);
    }
};
} }
