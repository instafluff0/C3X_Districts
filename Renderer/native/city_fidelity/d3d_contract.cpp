#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstring>
int main() {
    ID3D11Device*device=nullptr;ID3D11DeviceContext*context=nullptr;
    D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;
    HRESULT hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,&level,1,D3D11_SDK_VERSION,&device,nullptr,&context);
    if(FAILED(hr))return 1;
    char const*entries[]={"VSNativeCity","VSNativeCityReflection","PSNativeCity","PSNativeCityEmission","PSNativeCityReflection","PSNativeCityReflectionEmission","CSPost"};
    unsigned compiled=0;
    for(unsigned i=0;i<7;i++){
        ID3DBlob*blob=nullptr,*errors=nullptr;
        hr=D3DCompileFromFile(i==6?L"city_fidelity/hdr_glow.hlsl":L"city_fidelity/city.hlsl",nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entries[i],i<2?"vs_5_0":i<6?"ps_5_0":"cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errors);
        if(errors){std::fprintf(stderr,"%s: %s\n",entries[i],static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}
        if(FAILED(hr)){context->Release();device->Release();return 1;}
        ID3D11VertexShader*vs=nullptr;ID3D11PixelShader*ps=nullptr;ID3D11ComputeShader*cs=nullptr;
        if(i<2)hr=device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vs);
        else if(i<6)hr=device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&ps);
        else hr=device->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs);
        if(SUCCEEDED(hr) && i==0){
            D3D11_INPUT_ELEMENT_DESC e[]={
                {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",1,DXGI_FORMAT_R32G32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",2,DXGI_FORMAT_R32_FLOAT,0,60,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",3,DXGI_FORMAT_R32G32B32_FLOAT,0,68,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",4,DXGI_FORMAT_R32G32B32_FLOAT,0,80,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",5,DXGI_FORMAT_R32G32B32_FLOAT,0,120,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",6,DXGI_FORMAT_R32G32_FLOAT,0,152,D3D11_INPUT_PER_VERTEX_DATA,0}};
            ID3D11InputLayout*layout=nullptr;
            hr=device->CreateInputLayout(e,9,blob->GetBufferPointer(),blob->GetBufferSize(),&layout);
            if(layout)layout->Release();
        }
        if(vs)vs->Release();if(ps)ps->Release();if(cs)cs->Release();blob->Release();
        if(FAILED(hr)){context->Release();device->Release();return 1;}
        compiled++;std::printf("PASS %s\n",entries[i]);std::fflush(stdout);
    }
    context->Release();device->Release();std::printf("PASS native city shader contracts=%u\n",compiled);return 0;
}
