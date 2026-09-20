#pragma once
// Worker-owned D3D11 scratch targets. Pixels enter production caches only after
// the common scene-linear MSAA resolve and display transfer have completed.
namespace c3x_renderer { namespace render_core {
struct LinearTarget {
    ID3D11Texture2D *color=nullptr,*resolved=nullptr,*depth_texture=nullptr;
    ID3D11RenderTargetView *target=nullptr;
    ID3D11ShaderResourceView *view=nullptr,*samples=nullptr,*depth_samples=nullptr;
    ID3D11DepthStencilView *depth=nullptr;
    UINT width=0,height=0;
    template<class T> void release(T*& p) { if(p) { p->Release(); p=nullptr; } }
    void reset() {
        release(depth_samples); release(samples); release(depth); release(depth_texture); release(view); release(target);
        release(resolved); release(color); width=height=0;
    }
    ~LinearTarget() { reset(); }
    void swap(LinearTarget& other){
        std::swap(color,other.color);std::swap(resolved,other.resolved);std::swap(depth_texture,other.depth_texture);
        std::swap(target,other.target);std::swap(view,other.view);std::swap(samples,other.samples);
        std::swap(depth_samples,other.depth_samples);std::swap(depth,other.depth);
        std::swap(width,other.width);std::swap(height,other.height);
    }
    bool ensure(ID3D11Device* device,UINT w,UINT h,bool sampleable=false,bool resolve=true) {
        if(target && width==w && height==h && (!sampleable || samples) && (!resolve || resolved)) return true;
        reset();
        UINT quality=0;
        if(FAILED(device->CheckMultisampleQualityLevels(DXGI_FORMAT_R16G16B16A16_FLOAT,4,&quality)) ||
            quality==0) return false;
        D3D11_TEXTURE2D_DESC d={}; d.Width=w; d.Height=h; d.MipLevels=d.ArraySize=1;
        d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT; d.SampleDesc.Count=4;
        d.BindFlags=D3D11_BIND_RENDER_TARGET|(sampleable?D3D11_BIND_SHADER_RESOURCE:0); d.Usage=D3D11_USAGE_DEFAULT;
        HRESULT hr=device->CreateTexture2D(&d,nullptr,&color);
        if(SUCCEEDED(hr)) hr=device->CreateRenderTargetView(color,nullptr,&target);
        if(SUCCEEDED(hr) && sampleable) hr=device->CreateShaderResourceView(color,nullptr,&samples);
        d.SampleDesc.Count=1; d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        if(SUCCEEDED(hr) && resolve) hr=device->CreateTexture2D(&d,nullptr,&resolved);
        if(SUCCEEDED(hr) && resolve) hr=device->CreateShaderResourceView(resolved,nullptr,&view);
        d.SampleDesc.Count=4; d.Format=sampleable?DXGI_FORMAT_R24G8_TYPELESS:DXGI_FORMAT_D24_UNORM_S8_UINT;
        d.BindFlags=D3D11_BIND_DEPTH_STENCIL|(sampleable?D3D11_BIND_SHADER_RESOURCE:0);
        if(SUCCEEDED(hr)) hr=device->CreateTexture2D(&d,nullptr,&depth_texture);
        D3D11_DEPTH_STENCIL_VIEW_DESC ds={};ds.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;ds.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2DMS;
        if(SUCCEEDED(hr)) hr=device->CreateDepthStencilView(depth_texture,&ds,&depth);
        D3D11_SHADER_RESOURCE_VIEW_DESC dv={};dv.Format=DXGI_FORMAT_R24_UNORM_X8_TYPELESS;dv.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2DMS;
        if(SUCCEEDED(hr) && sampleable) hr=device->CreateShaderResourceView(depth_texture,&dv,&depth_samples);
        if(FAILED(hr)) { reset(); return false; }
        width=w; height=h; return true;
    }
    std::size_t bytes() const { return std::size_t(width)*height*(8*4+(resolved?8:0)+4*4); }
};
// Sample-preserving translation of a resident static scene. Newly exposed or
// invalidated rectangles are cleared in the same draw, before selected geometry.
struct LinearRestore {
    ID3D11VertexShader* vertex=nullptr;ID3D11PixelShader* pixel=nullptr;
    ID3D11Buffer* settings=nullptr;ID3D11DepthStencilState* depth=nullptr;
    ID3D11RasterizerState* rasterizer=nullptr;
    template<class T>void release(T*& p){if(p)p->Release();p=nullptr;}
    void reset(){release(vertex);release(pixel);release(settings);release(depth);release(rasterizer);}
    ~LinearRestore(){reset();}
    bool ensure(ID3D11Device* device){
        if(pixel)return true;
        char const* source=R"(
Texture2DMS<float4,4> scene:register(t0);
Texture2DMS<float,4> scene_depth:register(t1);
cbuffer Restore:register(b0){int4 move_extent;int4 metadata;int4 dirty[16];};
float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
struct Output{float4 color:SV_Target;float depth:SV_Depth;};
Output PS(float4 position:SV_Position,uint sample:SV_SampleIndex){
 int2 destination=int2(position.xy),p=destination;
 if(metadata.w!=0)p=p*2/metadata.w;
 p-=move_extent.xy;
 if(metadata.y!=0)p=(p%move_extent.zw+move_extent.zw)%move_extent.zw;
 bool valid=all(p>=0)&&all(p<move_extent.zw);
 for(int i=0;i<metadata.x;++i)if(all(destination>=dirty[i].xy*2)&&all(destination<dirty[i].zw*2))valid=false;
 Output result;result.color=valid?scene.Load(p,sample):float4(0,0,0,0);
 result.depth=valid?scene_depth.Load(p,sample):1;
 if(metadata.z!=0){result.color=0;result.depth=1;}return result;
})";
        auto compile=[&](char const* entry,char const* target,ID3DBlob** blob){
            ID3DBlob* errors=nullptr;HRESULT hr=D3DCompile(source,std::strlen(source),"scene_restore",nullptr,nullptr,
                entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,blob,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}return hr;
        };
        ID3DBlob* blob=nullptr;HRESULT hr=compile("VS","vs_5_0",&blob);
        if(SUCCEEDED(hr))hr=device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vertex);release(blob);
        if(SUCCEEDED(hr))hr=compile("PS","ps_5_0",&blob);
        if(SUCCEEDED(hr))hr=device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&pixel);release(blob);
        D3D11_BUFFER_DESC b={};b.ByteWidth=288;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&settings);
        D3D11_DEPTH_STENCIL_DESC d={};d.DepthEnable=true;d.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;d.DepthFunc=D3D11_COMPARISON_ALWAYS;
        if(SUCCEEDED(hr))hr=device->CreateDepthStencilState(&d,&depth);
        D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=true;r.MultisampleEnable=true;r.ScissorEnable=true;
        if(SUCCEEDED(hr))hr=device->CreateRasterizerState(&r,&rasterizer);
        if(FAILED(hr)){reset();return false;}return true;
    }
    bool draw(ID3D11DeviceContext* context,LinearTarget const& target,
              ID3D11ShaderResourceView* color,ID3D11ShaderResourceView* old_depth,
              int dx,int dy,std::vector<D3D11_RECT> const& dirty,
              std::vector<D3D11_RECT> const* regions=nullptr,unsigned source_width=0,unsigned source_height=0,bool circular=false,bool clear=false,int sample_scale=0,D3D11_RECT const* clip=nullptr,int offset_scale=2){
        if(dirty.size()>16)return false;
        struct Constants{int move_extent[4],metadata[4];D3D11_RECT dirty[16];} values={};
        values.move_extent[0]=dx*offset_scale;values.move_extent[1]=dy*offset_scale;
        values.move_extent[2]=int(source_width?source_width:target.width);values.move_extent[3]=int(source_height?source_height:target.height);values.metadata[0]=int(dirty.size());values.metadata[1]=circular?1:0;values.metadata[2]=clear?1:0;values.metadata[3]=sample_scale;
        std::copy(dirty.begin(),dirty.end(),values.dirty);context->UpdateSubresource(settings,0,nullptr,&values,0,0);
        context->OMSetRenderTargets(1,&target.target,target.depth);context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(depth,0);context->RSSetState(rasterizer);
        D3D11_VIEWPORT viewport={0,0,float(target.width),float(target.height),0,1};context->RSSetViewports(1,&viewport);
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);context->PSSetConstantBuffers(0,1,&settings);
        ID3D11ShaderResourceView* inputs[]={color,old_depth};context->PSSetShaderResources(0,2,inputs);
        auto draw=[&](D3D11_RECT rect){context->RSSetScissorRects(1,&rect);context->Draw(3,0);};
        if(clip)draw(*clip);
        else if(regions)for(auto rect:*regions){rect.left*=2;rect.top*=2;rect.right*=2;rect.bottom*=2;draw(rect);}
        else draw({0,0,LONG(target.width),LONG(target.height)});
        inputs[0]=inputs[1]=nullptr;context->PSSetShaderResources(0,2,inputs);context->OMSetRenderTargets(0,nullptr,nullptr);return true;
    }
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
              ID3D11ShaderResourceView* reconstructed=nullptr,UINT reconstructed_width=0,UINT reconstructed_height=0,
              D3D11_RECT const* dirty=nullptr) {
        context->OMSetRenderTargets(0,nullptr,nullptr);
        if(!reconstructed)context->ResolveSubresource(linear.resolved,0,linear.color,0,DXGI_FORMAT_R16G16B16A16_FLOAT);
        context->OMSetRenderTargets(1,&destination,nullptr);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(nullptr,0);
        UINT width=reconstructed?reconstructed_width:linear.width/render_scale,height=reconstructed?reconstructed_height:linear.height/render_scale;
        D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};
        D3D11_RECT rect={0,0,LONG(width),LONG(height)};
        if(dirty)rect=*dirty;
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
