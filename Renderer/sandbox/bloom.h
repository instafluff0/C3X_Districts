#pragma once

// Half-resolution HDR highlight response. The scene is resolved once, then
// extracted and blurred in two small targets before final backbuffer shading.
struct SandboxBloom {
    ID3D11Texture2D* color[2]={};
    ID3D11RenderTargetView* target[2]={};
    ID3D11ShaderResourceView* view[2]={};
    ID3D11VertexShader* vertex=nullptr;
    ID3D11PixelShader* extract=nullptr;
    ID3D11PixelShader* blur=nullptr;
    ID3D11Buffer* settings=nullptr;
    ID3D11SamplerState* sampler=nullptr;
    ID3D11RasterizerState* rasterizer=nullptr;
    unsigned width=0,height=0;
    template<class T> static void drop(T*& value){if(value)value->Release();value=nullptr;}
    ~SandboxBloom(){reset();}
    void reset(){
        for(unsigned i=0;i<2;++i){drop(view[i]);drop(target[i]);drop(color[i]);}
        drop(vertex);drop(extract);drop(blur);drop(settings);drop(sampler);drop(rasterizer);
        width=height=0;
    }
    bool ensure(unsigned source_width,unsigned source_height){
        unsigned w=(source_width+1)/2,h=(source_height+1)/2;
        if(view[0] && width==w && height==h)return true;
        reset();width=w;height=h;
        char const* source=R"(
Texture2D<float4> input_image:register(t0);
Texture2D<float4> dynamic_image:register(t1);
SamplerState linear_clamp:register(s0);
cbuffer BloomSettings:register(b0){float4 dimensions_axis;};
float4 VS(uint id:SV_VertexID):SV_Position{
 float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);
}
float3 highlight(float4 c){
 if(c.a<=0)return 0;
 float peak=max(c.r,max(c.g,c.b));
 // Leave sunlit terrain below the bloom knee; preserve strong HDR highlights.
 return c.rgb*(max(peak-2.5,0)/max(peak,.0001));
}
float4 PSExtract(float4 position:SV_Position):SV_Target{
 int2 at=int2(position.xy)*2;
 float3 c=0;
 [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x)
 {
  int3 sample_at=int3(at+int2(x,y),0);
  float4 base=input_image.Load(sample_at);
  float4 moving=dynamic_image.Load(sample_at);
  c+=highlight(moving+base*(1-moving.a));
 }
 return float4(c*.25,1);
}
float4 PSBlur(float4 position:SV_Position):SV_Target{
 float2 uv=(position.xy+.5)/dimensions_axis.xy;
 float2 delta=dimensions_axis.zw/dimensions_axis.xy;
 float3 c=input_image.SampleLevel(linear_clamp,uv,0).rgb*.227027;
 [unroll]for(int i=1;i<=4;++i){
  float weight=i==1?.1945946:i==2?.1216216:i==3?.0540541:.0162162;
  c+=weight*(input_image.SampleLevel(linear_clamp,uv+delta*i,0).rgb+
             input_image.SampleLevel(linear_clamp,uv-delta*i,0).rgb);
 }
 return float4(c,1);
})";
        auto compile=[&](char const* entry,char const* profile,ID3DBlob** result){
            ID3DBlob* errors=nullptr;
            HRESULT hr=D3DCompile(source,std::strlen(source),"sandbox_bloom",nullptr,nullptr,
                entry,profile,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,result,&errors);
            if(errors){if(FAILED(hr))std::printf("SANDBOX_BLOOM_SHADER %s\n",
                static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
            return hr;
        };
        HRESULT hr=S_OK;
        ID3DBlob* code=nullptr;
        if(SUCCEEDED(hr))hr=compile("VS","vs_5_0",&code);
        if(SUCCEEDED(hr))hr=renderer.device->CreateVertexShader(code->GetBufferPointer(),
            code->GetBufferSize(),nullptr,&vertex);drop(code);
        if(SUCCEEDED(hr))hr=compile("PSExtract","ps_5_0",&code);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(code->GetBufferPointer(),
            code->GetBufferSize(),nullptr,&extract);drop(code);
        if(SUCCEEDED(hr))hr=compile("PSBlur","ps_5_0",&code);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(code->GetBufferPointer(),
            code->GetBufferSize(),nullptr,&blur);drop(code);
        D3D11_BUFFER_DESC buffer{};buffer.ByteWidth=16;buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&buffer,nullptr,&settings);
        D3D11_SAMPLER_DESC filter{};filter.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        filter.AddressU=filter.AddressV=filter.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;
        filter.MaxLOD=D3D11_FLOAT32_MAX;
        if(SUCCEEDED(hr))hr=renderer.device->CreateSamplerState(&filter,&sampler);
        D3D11_RASTERIZER_DESC raster{};raster.FillMode=D3D11_FILL_SOLID;
        raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=true;
        if(SUCCEEDED(hr))hr=renderer.device->CreateRasterizerState(&raster,&rasterizer);
        D3D11_TEXTURE2D_DESC texture{};texture.Width=w;texture.Height=h;
        texture.ArraySize=texture.MipLevels=texture.SampleDesc.Count=1;
        texture.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        texture.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        for(unsigned i=0;i<2 && SUCCEEDED(hr);++i){
            hr=renderer.device->CreateTexture2D(&texture,nullptr,&color[i]);
            if(SUCCEEDED(hr))hr=renderer.device->CreateRenderTargetView(color[i],nullptr,&target[i]);
            if(SUCCEEDED(hr))hr=renderer.device->CreateShaderResourceView(color[i],nullptr,&view[i]);
        }
        if(FAILED(hr)){reset();return false;}
        return true;
    }
    bool draw(ID3D11ShaderResourceView* static_resolved,
            ID3D11ShaderResourceView* dynamic_resolved){
        if(!static_resolved || !dynamic_resolved || !view[0])return false;
        auto* context=renderer.context;
        context->OMSetDepthStencilState(nullptr,0);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->RSSetState(rasterizer);
        D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};
        context->RSSetViewports(1,&viewport);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);
        context->PSSetConstantBuffers(0,1,&settings);
        context->PSSetSamplers(0,1,&sampler);
        auto pass=[&](ID3D11RenderTargetView* output,ID3D11ShaderResourceView* input,
                ID3D11ShaderResourceView* moving,ID3D11PixelShader* shader,float x,float y){
            float values[]={float(width),float(height),x,y};
            context->UpdateSubresource(settings,0,nullptr,values,0,0);
            context->OMSetRenderTargets(1,&output,nullptr);
            context->PSSetShader(shader,nullptr,0);
            ID3D11ShaderResourceView* inputs[]={input,moving};
            context->PSSetShaderResources(0,2,inputs);
            context->Draw(3,0);
            ID3D11ShaderResourceView* empty[]={nullptr,nullptr};
            context->PSSetShaderResources(0,2,empty);
        };
        pass(target[0],static_resolved,dynamic_resolved,extract,0,0);
        pass(target[1],view[0],nullptr,blur,1,0);
        pass(target[0],view[1],nullptr,blur,0,1);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }
    std::size_t bytes() const{return std::size_t(width)*height*16;}
};
