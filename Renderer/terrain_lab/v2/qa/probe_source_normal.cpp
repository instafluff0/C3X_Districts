// Standalone licensed-source compute probe. Never loads or modifies the game.
#include <d3d11.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

void checked(HRESULT hr) { if(FAILED(hr)) throw std::runtime_error("D3D probe failure "+std::to_string(hr)); }
std::vector<char> read_file(const char* path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f)throw std::runtime_error("missing probe input");
    std::vector<char> data(size_t(f.tellg()));f.seekg(0);f.read(data.data(),data.size());return data;
}
int main() {
    try {
        auto code=read_file("shader-00426326.dxbc"), height=read_file("normal-probe-input.f32");
        if(height.size()!=34*34*4)throw std::runtime_error("incorrect input dimensions");
        ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;
        D3D_FEATURE_LEVEL requested=D3D_FEATURE_LEVEL_11_0, actual;
        checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,&requested,1,D3D11_SDK_VERSION,&device,&actual,&context));
        ID3D11ComputeShader* shader=nullptr;
        checked(device->CreateComputeShader(code.data(),code.size(),nullptr,&shader));
        D3D11_TEXTURE2D_DESC desc={};desc.Width=34;desc.Height=34;desc.MipLevels=1;desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_R32_FLOAT;desc.SampleDesc.Count=1;desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data={height.data(),34*4,0};ID3D11Texture2D* input=nullptr;
        checked(device->CreateTexture2D(&desc,&data,&input));ID3D11ShaderResourceView* srv=nullptr;
        checked(device->CreateShaderResourceView(input,nullptr,&srv));
        desc.Width=32;desc.Height=32;desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
        ID3D11Texture2D* output=nullptr;checked(device->CreateTexture2D(&desc,nullptr,&output));
        ID3D11UnorderedAccessView* uav=nullptr;checked(device->CreateUnorderedAccessView(output,nullptr,&uav));
        float scale[4]={1,0,0,0};unsigned globals[8]={};globals[6]=1;
        D3D11_BUFFER_DESC bd={};bd.ByteWidth=16;bd.Usage=D3D11_USAGE_IMMUTABLE;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA sd={scale,0,0};ID3D11Buffer* buffers[2]={};
        checked(device->CreateBuffer(&bd,&sd,&buffers[0]));bd.ByteWidth=32;sd.pSysMem=globals;
        checked(device->CreateBuffer(&bd,&sd,&buffers[1]));
        context->CSSetShader(shader,nullptr,0);context->CSSetConstantBuffers(0,2,buffers);
        context->CSSetShaderResources(2,1,&srv);context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);
        context->Dispatch(2,2,1);
        desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging=nullptr;checked(device->CreateTexture2D(&desc,nullptr,&staging));
        context->CopyResource(staging,output);D3D11_MAPPED_SUBRESOURCE mapped={};checked(context->Map(staging,0,D3D11_MAP_READ,0,&mapped));
        std::ofstream f("normal-probe-output.rgba8",std::ios::binary);
        for(unsigned y=0;y<32;++y)f.write(static_cast<char*>(mapped.pData)+y*mapped.RowPitch,128);
        context->Unmap(staging,0);f.close();if(!f)throw std::runtime_error("probe output write failed");
        std::cout<<"PASS source normal compute dispatch: 1024 pixels, WARP, scale 1\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
