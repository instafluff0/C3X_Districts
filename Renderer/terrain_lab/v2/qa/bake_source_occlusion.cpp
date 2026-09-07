// Offline local-source compute execution. Output is generic derived material data.
#include <d3d11.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

void checked(HRESULT hr) {if(FAILED(hr))throw std::runtime_error(std::to_string(hr));}
std::vector<char> bytes(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("missing "+path);
    std::vector<char> b(size_t(f.tellg()));f.seekg(0);f.read(b.data(),b.size());return b;
}
ID3D11ShaderResourceView* field(ID3D11Device* device,const void* data,unsigned side) {
    D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=side;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
    d.Format=DXGI_FORMAT_R32_FLOAT;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA s={data,side*4,0};ID3D11Texture2D* t=nullptr;
    checked(device->CreateTexture2D(&d,&s,&t));ID3D11ShaderResourceView* view=nullptr;
    checked(device->CreateShaderResourceView(t,nullptr,&view));t->Release();return view;
}
int main() {
    try {
        ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;
        D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0,actual;
        checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,&level,1,D3D11_SDK_VERSION,&device,&actual,&context));
        auto code=bytes("shader-00426f26.dxbc");ID3D11ComputeShader* shader=nullptr;
        checked(device->CreateComputeShader(code.data(),code.size(),nullptr,&shader));
        context->CSSetShader(shader,nullptr,0);
        float scale[4]={1,0,0,0};unsigned globals[8]={};globals[6]=8;
        D3D11_BUFFER_DESC bd={};bd.ByteWidth=16;bd.Usage=D3D11_USAGE_IMMUTABLE;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA sd={scale,0,0};ID3D11Buffer* buffers[2]={};
        checked(device->CreateBuffer(&bd,&sd,&buffers[0]));bd.ByteWidth=32;sd.pSysMem=globals;
        checked(device->CreateBuffer(&bd,&sd,&buffers[1]));context->CSSetConstantBuffers(0,2,buffers);
        std::vector<float> spec(1024*1024,62.0f/255.0f);
        for(const char* name:{"grassland","plains","desert","marsh","tundra","grassland_shift","flat_control"}) {
            auto high=bytes(std::string(name)+"-ao-high.f32"),half=bytes(std::string(name)+"-ao-half.f32");
            if(high.size()!=1040*1040*4||half.size()!=520*520*4)throw std::runtime_error("input dimensions");
            ID3D11ShaderResourceView* views[4]={field(device,half.data(),520),nullptr,field(device,high.data(),1040),field(device,spec.data(),1024)};
            context->CSSetShaderResources(0,4,views);
            D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=1024;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
            d.Format=DXGI_FORMAT_R8G8B8A8_UNORM;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
            ID3D11Texture2D* output=nullptr;checked(device->CreateTexture2D(&d,nullptr,&output));
            ID3D11UnorderedAccessView* uav=nullptr;checked(device->CreateUnorderedAccessView(output,nullptr,&uav));
            context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);context->Dispatch(128,64,1);
            d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            ID3D11Texture2D* staging=nullptr;checked(device->CreateTexture2D(&d,nullptr,&staging));context->CopyResource(staging,output);
            D3D11_MAPPED_SUBRESOURCE mapped={};checked(context->Map(staging,0,D3D11_MAP_READ,0,&mapped));
            std::ofstream f(std::string(name)+"-ao.rgba8",std::ios::binary);
            for(unsigned y=0;y<1024;++y)f.write(static_cast<char*>(mapped.pData)+y*mapped.RowPitch,4096);
            context->Unmap(staging,0);f.close();if(!f)throw std::runtime_error("output write");
            for(auto* view:views)if(view)view->Release();uav->Release();output->Release();staging->Release();
            std::cout<<"PASS original source occlusion compute: "<<name<<"\n";
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL source occlusion: "<<e.what()<<"\n";return 1;}
}
