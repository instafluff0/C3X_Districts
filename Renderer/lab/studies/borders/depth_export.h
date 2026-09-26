#pragma once
// Opt-in Lab readback of the depth actually used by the city renderer. The
// production pass remains unchanged; this reads each finished MSAA region.
#include <cstdio>
#include <string>
#include <vector>

namespace c3x_renderer { namespace fidelity {
struct BorderDepthExport {
    ID3D11VertexShader* vertex=nullptr;
    ID3D11PixelShader* pixel=nullptr;
    ID3D11Texture2D* image=nullptr;
    ID3D11RenderTargetView* target=nullptr;
    ID3D11Texture2D* staging=nullptr;
    UINT width=0,height=0;
    template<class T> void release(T*& object){if(object){object->Release();object=nullptr;}}
    void reset(){release(staging);release(target);release(image);release(pixel);release(vertex);width=height=0;}
    ~BorderDepthExport(){reset();}

    bool ensure(ID3D11Device* device,UINT w,UINT h){
        if(!pixel){
            char const* source=R"(
Texture2DMS<float,4> scene_depth:register(t0);
float4 VS(uint id:SV_VertexID):SV_Position{
 float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);
}
float PS(float4 position:SV_Position):SV_Target{
 int2 p=int2(position.xy)*2;float nearest=1;
 [unroll]for(int y=0;y<2;y++)[unroll]for(int x=0;x<2;x++)
  [unroll]for(int sample=0;sample<4;sample++)
   nearest=min(nearest,scene_depth.Load(p+int2(x,y),sample));
 return nearest;
})";
            ID3DBlob* blob=nullptr;ID3DBlob* errors=nullptr;
            HRESULT hr=D3DCompile(source,std::strlen(source),"border_depth",nullptr,nullptr,
                "VS","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}
            if(SUCCEEDED(hr))hr=device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vertex);
            release(blob);
            if(SUCCEEDED(hr))hr=D3DCompile(source,std::strlen(source),"border_depth",nullptr,nullptr,
                "PS","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}
            if(SUCCEEDED(hr))hr=device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&pixel);
            release(blob);
            if(FAILED(hr))return false;
        }
        if(target && width==w && height==h)return true;
        release(staging);release(target);release(image);width=height=0;
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;
        desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_R32_FLOAT;desc.BindFlags=D3D11_BIND_RENDER_TARGET;
        HRESULT hr=device->CreateTexture2D(&desc,nullptr,&image);
        if(SUCCEEDED(hr))hr=device->CreateRenderTargetView(image,nullptr,&target);
        desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        if(SUCCEEDED(hr))hr=device->CreateTexture2D(&desc,nullptr,&staging);
        if(FAILED(hr))return false;
        width=w;height=h;return true;
    }

    bool save(ID3D11Device* device,ID3D11DeviceContext* context,
              ID3D11ShaderResourceView* scene_depth,UINT w,UINT h,
              std::string const& prefix,int left,int top,int right,int bottom,
              int source_x,int source_y,float depth_offset){
        if(!scene_depth || !ensure(device,w,h))return false;
        context->OMSetRenderTargets(1,&target,nullptr);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(nullptr,0);
        D3D11_VIEWPORT viewport={0,0,float(w),float(h),0,1};context->RSSetViewports(1,&viewport);
        D3D11_RECT scissor={0,0,LONG(w),LONG(h)};context->RSSetScissorRects(1,&scissor);
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->PSSetShaderResources(0,1,&scene_depth);context->Draw(3,0);
        ID3D11ShaderResourceView* empty=nullptr;context->PSSetShaderResources(0,1,&empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->CopyResource(staging,image);
        D3D11_MAPPED_SUBRESOURCE mapped={};
        if(FAILED(context->Map(staging,0,D3D11_MAP_READ,0,&mapped)))return false;
        std::string path=prefix+".depth."+std::to_string(left)+"_"+std::to_string(top)+".bin";
        std::FILE* file=nullptr;bool ok=!fopen_s(&file,path.c_str(),"wb") && file;
        if(ok){
            int rectangle[]={left,top,right-left,bottom-top};
            ok=std::fwrite("C3XBDP1\0",1,8,file)==8 &&
               std::fwrite(rectangle,sizeof(rectangle),1,file)==1 &&
               std::fwrite(&depth_offset,sizeof(depth_offset),1,file)==1;
            for(int y=top;y<bottom && ok;++y){
                auto row=static_cast<char const*>(mapped.pData)+
                    std::size_t(source_y+y-top)*mapped.RowPitch+std::size_t(source_x)*sizeof(float);
                ok=std::fwrite(row,sizeof(float),std::size_t(right-left),file)==std::size_t(right-left);
            }
            if(std::fclose(file)!=0)ok=false;
        }
        context->Unmap(staging,0);
        return ok;
    }
};
}}
