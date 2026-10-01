#pragma once
// Private, bounded diagnostics for the existing underlay implementation.
// No readback/state creation runs when the diagnostic environment is empty.
struct SandboxUnderlayRejectionProbe {
    bool configured=false,active=false,done[3]={};
    unsigned point=0;
    int control=0;
    std::string directory;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> forced;
    void configure(){
        if(configured)return;configured=true;
        char value[4096]{};
        if(GetEnvironmentVariableA("C3X_UNDERLAY_MASK_DIR",value,sizeof(value)))directory=value;
        value[0]=0;
        if(GetEnvironmentVariableA("C3X_UNDERLAY_STENCIL_CONTROL",value,sizeof(value)))control=std::atoi(value);
    }
    void begin(bool is_static,float zoom){
        configure();active=false;
        if(!is_static || directory.empty())return;
        float points[]={1,1+22.f/180,1.25f};
        for(unsigned i=0;i<3;++i)if(!done[i] && std::abs(zoom-points[i])<.0000005f){
            done[i]=true;point=i;active=true;
            std::printf("UNDERLAY_MASK_BEGIN point=%u zoom=%.9f\n",point,zoom);break;
        }
    }
    bool control_state(ID3D11Device* device,ID3D11DeviceContext* context,
                       ID3D11DepthStencilState* original){
        configure();
        if(control!=1 && control!=2)return true;
        if(!forced){
            D3D11_DEPTH_STENCIL_DESC desc{};original->GetDesc(&desc);
            desc.FrontFace.StencilFunc=control==1?D3D11_COMPARISON_ALWAYS:D3D11_COMPARISON_NEVER;
            desc.BackFace=desc.FrontFace;
            if(FAILED(device->CreateDepthStencilState(&desc,&forced)))return false;
            std::printf("UNDERLAY_STENCIL_CONTROL mode=%d compare=%u\n",control,unsigned(desc.FrontFace.StencilFunc));
        }
        context->OMSetDepthStencilState(forced.Get(),0);return true;
    }
    void bound(ID3D11DeviceContext* context,unsigned stage,unsigned layer,
               ID3D11PixelShader* expected,float zoom){
        if(!active)return;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilState> state;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depth;
        Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel;
        Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex;
        Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
        Microsoft::WRL::ComPtr<ID3D11Resource> resource;
        UINT reference=0,stride=0,offset=0;
        context->OMGetDepthStencilState(&state,&reference);
        context->OMGetRenderTargets(0,nullptr,&depth);
        context->PSGetShader(&pixel,nullptr,nullptr);context->VSGetShader(&vertex,nullptr,nullptr);
        context->IAGetVertexBuffers(0,1,&buffer,&stride,&offset);
        if(depth)depth->GetResource(&resource);
        D3D11_DEPTH_STENCIL_DESC desc{};if(state)state->GetDesc(&desc);
        D3D11_VIEWPORT viewport{};UINT count=1;context->RSGetViewports(&count,&viewport);
        D3D11_RECT scissor{};count=1;context->RSGetScissorRects(&count,&scissor);
        std::printf("UNDERLAY_BOUND point=%u stage=%u layer=%u zoom=%.9f dsv=%p resource=%p state=%p depth=%u write=%u func=%u stencil=%u read=%u write_mask=%u front=%u back=%u pass=%u ref=%u ps=%p expected=%p ps_match=%u vs=%p stride=%u offset=%u viewport=%.3f,%.3f,%.3f,%.3f,%.6f,%.6f scissor=%ld,%ld,%ld,%ld\n",
            point,stage,layer,zoom,static_cast<void*>(depth.Get()),static_cast<void*>(resource.Get()),static_cast<void*>(state.Get()),
            unsigned(desc.DepthEnable),unsigned(desc.DepthWriteMask),unsigned(desc.DepthFunc),unsigned(desc.StencilEnable),
            unsigned(desc.StencilReadMask),unsigned(desc.StencilWriteMask),unsigned(desc.FrontFace.StencilFunc),
            unsigned(desc.BackFace.StencilFunc),unsigned(desc.FrontFace.StencilPassOp),reference,
            static_cast<void*>(pixel.Get()),static_cast<void*>(expected),unsigned(pixel.Get()==expected),static_cast<void*>(vertex.Get()),stride,offset,
            viewport.TopLeftX,viewport.TopLeftY,viewport.Width,viewport.Height,viewport.MinDepth,viewport.MaxDepth,
            scissor.left,scissor.top,scissor.right,scissor.bottom);
    }
    bool copy(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11View* view,char const* phase){
        if(!active)return true;
        Microsoft::WRL::ComPtr<ID3D11Resource> resource;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        view->GetResource(&resource);
        if(FAILED(resource.As(&texture)))return false;
        D3D11_TEXTURE2D_DESC desc{};texture->GetDesc(&desc);
        if(desc.SampleDesc.Count!=1 || desc.ArraySize!=1 || desc.MipLevels!=1)return false;
        unsigned bytes=desc.Format==DXGI_FORMAT_R24G8_TYPELESS?4:
            desc.Format==DXGI_FORMAT_R16G16B16A16_FLOAT?8:0;
        if(!bytes)return false;
        D3D11_TEXTURE2D_DESC previous{};if(staging)staging->GetDesc(&previous);
        if(!staging || previous.Width!=desc.Width || previous.Height!=desc.Height || previous.Format!=desc.Format){
            staging.Reset();auto read=desc;read.Usage=D3D11_USAGE_STAGING;read.BindFlags=0;
            read.CPUAccessFlags=D3D11_CPU_ACCESS_READ;read.MiscFlags=0;
            if(FAILED(device->CreateTexture2D(&read,nullptr,&staging)))return false;
        }
        // Preserve every output binding across the diagnostic copy/map.
        ID3D11RenderTargetView* targets[8]={};ID3D11DepthStencilView* depth=nullptr;
        context->OMGetRenderTargets(8,targets,&depth);context->OMSetRenderTargets(0,nullptr,nullptr);
        context->CopyResource(staging.Get(),texture.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        HRESULT result=context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped);
        bool ok=SUCCEEDED(result);
        if(ok){
            char file[4096];sprintf_s(file,"%s/mask-z%u.%s.raw",directory.c_str(),point,phase);
            FILE* output=nullptr;ok=fopen_s(&output,file,"wb")==0 && output;
            if(ok){
                unsigned header[]={desc.Width,desc.Height,unsigned(desc.Format)};
                ok=std::fwrite(header,sizeof(header),1,output)==1;
                for(unsigned y=0;y<desc.Height && ok;++y)
                    ok=std::fwrite(static_cast<unsigned char*>(mapped.pData)+y*mapped.RowPitch,
                        std::size_t(desc.Width)*bytes,1,output)==1;
                ok=std::fclose(output)==0 && ok;
            }
            context->Unmap(staging.Get(),0);
        }
        context->OMSetRenderTargets(8,targets,depth);
        for(auto* target:targets)if(target)target->Release();if(depth)depth->Release();
        std::printf("UNDERLAY_COPY point=%u phase=%s resource=%p width=%u height=%u format=%u sample=%u bytes=%u result=%u\n",
            point,phase,static_cast<void*>(resource.Get()),desc.Width,desc.Height,unsigned(desc.Format),desc.SampleDesc.Count,bytes,unsigned(ok));
        return ok;
    }
};
