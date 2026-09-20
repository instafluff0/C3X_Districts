// Independent D3D witness: region-produced static depth is consumed by one
// view-wide dynamic draw. Uses the production vertex adapter and D24/MSAA4.
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include "render_core/draw_parameter_stream.h"
#include "render_core/immutable_mesh_upload.h"

int main(){
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::puts("depth-contract: device setup");
    assert(SetCurrentDirectoryW(L"../../../..")); // native_cpp_test's isolated build directory
    auto create=reinterpret_cast<decltype(&D3D11CreateDevice)>(GetProcAddress(LoadLibraryA("d3d11.dll"),"D3D11CreateDevice"));
    auto compile_file=reinterpret_cast<decltype(&D3DCompileFromFile)>(GetProcAddress(LoadLibraryA("d3dcompiler_47.dll"),"D3DCompileFromFile"));
    auto compile=reinterpret_cast<decltype(&D3DCompile)>(GetProcAddress(GetModuleHandleA("d3dcompiler_47.dll"),"D3DCompile"));
    assert(create && compile_file && compile);
    ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;
    D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;
    assert(SUCCEEDED(create(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,&level,1,D3D11_SDK_VERSION,&device,nullptr,&context)));
    std::puts("depth-contract: production shader");
    ID3DBlob* blob=nullptr;ID3DBlob* errors=nullptr;
    auto hr=compile_file(L"Renderer/native/city_fidelity/feature.hlsl",nullptr,nullptr,"VSIntegratedFeature","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errors);
    if(errors){std::printf("%s",static_cast<char*>(errors->GetBufferPointer()));errors->Release();errors=nullptr;}assert(SUCCEEDED(hr));
    ID3D11VertexShader* vs=nullptr;assert(SUCCEEDED(device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vs)));
    D3D11_INPUT_ELEMENT_DESC elements[]={
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,20,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",6,DXGI_FORMAT_R32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",14,DXGI_FORMAT_R32G32B32_FLOAT,0,36,D3D11_INPUT_PER_VERTEX_DATA,0}};
    ID3D11InputLayout* layout=nullptr;assert(SUCCEEDED(device->CreateInputLayout(elements,UINT(sizeof(elements)/sizeof(elements[0])),blob->GetBufferPointer(),blob->GetBufferSize(),&layout)));blob->Release();blob=nullptr;
    char const* ps_source="cbuffer Color:register(b0){float4 color;}float4 PS():SV_Target{return color;}";
    assert(SUCCEEDED(compile(ps_source,std::strlen(ps_source),"depth-witness",nullptr,nullptr,"PS","ps_5_0",0,0,&blob,&errors)));
    ID3D11PixelShader* ps=nullptr;assert(SUCCEEDED(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&ps)));blob->Release();
    std::puts("depth-contract: targets");
    D3D11_TEXTURE2D_DESC td={};td.Width=td.Height=128;td.ArraySize=td.MipLevels=1;
    td.SampleDesc.Count=4;td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;td.BindFlags=D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D *color=nullptr,*depth=nullptr,*resolved=nullptr,*readback=nullptr;
    assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&color)));
    ID3D11RenderTargetView* target=nullptr;assert(SUCCEEDED(device->CreateRenderTargetView(color,nullptr,&target)));
    td.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;td.BindFlags=D3D11_BIND_DEPTH_STENCIL;
    assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&depth)));
    ID3D11DepthStencilView* dsv=nullptr;assert(SUCCEEDED(device->CreateDepthStencilView(depth,nullptr,&dsv)));
    td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;td.SampleDesc.Count=1;td.BindFlags=0;
    assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&resolved)));
    td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&readback)));
    D3D11_DEPTH_STENCIL_DESC ds={};ds.DepthEnable=true;ds.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;ds.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
    ID3D11DepthStencilState* depth_state=nullptr;assert(SUCCEEDED(device->CreateDepthStencilState(&ds,&depth_state)));
    D3D11_RASTERIZER_DESC rs={};rs.FillMode=D3D11_FILL_SOLID;rs.CullMode=D3D11_CULL_NONE;rs.DepthClipEnable=true;rs.MultisampleEnable=true;
    ID3D11RasterizerState* rasterizer=nullptr;assert(SUCCEEDED(device->CreateRasterizerState(&rs,&rasterizer)));
    D3D11_BUFFER_DESC bd={};bd.ByteWidth=48;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    ID3D11Buffer *viewport_buffer=nullptr,*color_buffer=nullptr,*vertices=nullptr;
    assert(SUCCEEDED(device->CreateBuffer(&bd,nullptr,&viewport_buffer)));
    bd.ByteWidth=16;assert(SUCCEEDED(device->CreateBuffer(&bd,nullptr,&color_buffer)));
    bd.ByteWidth=6*48;bd.BindFlags=D3D11_BIND_VERTEX_BUFFER;
    assert(SUCCEEDED(device->CreateBuffer(&bd,nullptr,&vertices)));
    context->VSSetShader(vs,nullptr,0);context->PSSetShader(ps,nullptr,0);context->IASetInputLayout(layout);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetConstantBuffers(1,1,&viewport_buffer);context->PSSetConstantBuffers(0,1,&color_buffer);
    UINT stride=48,offset=0;context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
    context->OMSetDepthStencilState(depth_state,0);context->RSSetState(rasterizer);
    float world_depth_offset=0;bool streamed=false,packed_mesh=false,narrow_mesh=false,short_parameters=false;unsigned parameter_index=0;
    c3x_renderer::render_core::DrawParameterStream parameters;
    auto view=[&](int top,int height,float depth_translation){
        float settings[12]={0,-float(top),depth_translation+world_depth_offset,0,1.f/128,1.f/height,128,0};
        if(streamed){
            std::array<std::array<float,12>,256> batch{};
            // Unused entries deliberately disagree. Exercise the final legal
            // range and DISCARD while previous commands still borrow the buffer.
            for(auto& record:batch)record[0]=10000;
            unsigned count=short_parameters?std::array<unsigned,4>{3,17,127,256}[parameter_index%4]:256;
            unsigned index=short_parameters?count-1:std::array<unsigned,3>{0,17,255}[parameter_index%3];
            ++parameter_index;
            std::memcpy(batch[index].data(),settings,sizeof(settings));
            assert(!parameters.upload(batch.data(),0) && !parameters.upload(batch.data(),257));
            assert(parameters.upload(batch.data(),count));parameters.bind(1,index);
        }else{
            context->VSSetConstantBuffers(1,1,&viewport_buffer);
            context->UpdateSubresource(viewport_buffer,0,nullptr,settings,0,0);
        }
        D3D11_VIEWPORT vp={0,float(top),128,float(height),0,1};context->RSSetViewports(1,&vp);
    };
    auto quad=[&](float left,float right,float z,bool green){
        std::array<std::array<float,12>,6> data={};
        float points[6][2]={{left,0},{right,0},{left,128},{left,128},{right,0},{right,128}};
        for(unsigned i=0;i<6;++i){data[i][0]=points[i][0];data[i][1]=points[i][1];data[i][2]=z+points[i][1]*.25f;data[i][7]=1;}
        ID3D11Buffer* allocation=nullptr;
        unsigned vertex_range=0,index_range=0;
        if(packed_mesh){
            c3x_renderer::render_core::ImmutableMeshUpload upload;
            // Deliberately unsuitable first range: drawing offset zero must
            // disagree with the independent ordinary color/depth witness.
            std::array<unsigned char,13> poison{};poison.fill(0xff);
            assert(upload.append(poison.data(),poison.size())==0);
            vertex_range=upload.append(data.data(),sizeof(data));
            std::uint16_t narrow_indices[]={0,1,2,3,4,5};unsigned wide_indices[]={0,1,2,3,4,5};
            index_range=narrow_mesh?upload.append(narrow_indices,sizeof(narrow_indices)):upload.append(wide_indices,sizeof(wide_indices));
            assert(vertex_range%4==0 && index_range%4==0);
            assert(upload.create(device,&allocation));
            context->IASetVertexBuffers(0,1,&allocation,&stride,&vertex_range);
            context->IASetIndexBuffer(allocation,narrow_mesh?DXGI_FORMAT_R16_UINT:DXGI_FORMAT_R32_UINT,index_range);
        }else{
            context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
            context->UpdateSubresource(vertices,0,nullptr,data.data(),0,0);
        }
        float tint[4]={green?0.f:1.f,green?1.f:0.f,0,1};context->UpdateSubresource(color_buffer,0,nullptr,tint,0,0);
        if(packed_mesh){context->DrawIndexed(6,0,0);allocation->Release();}
        else context->Draw(6,0);
    };
    auto render=[&](bool common_consumer,bool legacy_depth,bool coplanar){
        float clear[4]={0,0,0,0};context->ClearRenderTargetView(target,clear);context->ClearDepthStencilView(dsv,D3D11_CLEAR_DEPTH,1,0);
        context->OMSetRenderTargets(1,&target,dsv);
        for(int top:{0,64}){
            view(top,64,legacy_depth?-float(top):0);quad(0,128,200,false);
            if(!common_consumer){quad(0,64,coplanar?200.f:180.f,true);quad(64,128,coplanar?200.f:220.f,true);}
        }
        if(common_consumer){view(0,128,0);quad(0,64,coplanar?200.f:180.f,true);quad(64,128,coplanar?200.f:220.f,true);}
        context->OMSetRenderTargets(0,nullptr,nullptr);context->ResolveSubresource(resolved,0,color,0,DXGI_FORMAT_R8G8B8A8_UNORM);context->CopyResource(readback,resolved);
        D3D11_MAPPED_SUBRESOURCE mapped={};assert(SUCCEEDED(context->Map(readback,0,D3D11_MAP_READ,0,&mapped)));
        std::vector<unsigned char> result(128*128*4);
        for(unsigned y=0;y<128;++y)std::memcpy(result.data()+y*512,static_cast<unsigned char*>(mapped.pData)+y*mapped.RowPitch,512);
        context->Unmap(readback,0);return result;
    };
    std::puts("depth-contract: common consumer");
    auto ordinary=render(false,false,false),shared=render(true,false,false),bad=render(true,true,false);
    assert(ordinary==shared && shared!=bad);
    assert(shared[32*512+32*4]==255 && shared[96*512+32*4]==255);
    assert(shared[32*512+96*4+1]==255 && shared[96*512+96*4+1]==255);
    auto coplanar=render(false,false,true);assert(coplanar==render(true,false,true));
    std::size_t changed=0;for(std::size_t i=0;i<bad.size();i+=4)changed+=std::memcmp(bad.data()+i,shared.data()+i,4)!=0;
    std::printf("PASS production common-depth consumer: exact occlusion and coplanar ordering; legacy-negative-control changed_pixels=%zu\n",changed);
    for(float origin:{-4096.f,-2048.f,-128.f,128.f,636.f,2048.f,4096.f}){
        world_depth_offset=origin;
        assert(render(false,false,false)==ordinary);
        assert(render(true,false,false)==ordinary);
        assert(render(false,false,true)==coplanar);
        assert(render(true,false,true)==coplanar);
    }
    std::puts("PASS common-depth occlusion and identical coplanar draws across seven origin shifts");
    if(parameters.available(device,context)){
        streamed=true;
        for(float origin:{-4096.f,-128.f,0.f,636.f,4096.f}){
            world_depth_offset=origin;
            assert(render(false,false,false)==ordinary && render(true,false,false)==ordinary);
            assert(render(false,false,true)==coplanar && render(true,false,true)==coplanar);
        }
        assert(parameters.uploads>32 && parameters.records==parameters.uploads*256);
        auto supported=parameters.no_overwrite;
        for(bool append:{false,true}){
            parameters.no_overwrite=append && supported;
            short_parameters=true;
            for(int repeat=0;repeat<12;++repeat){
                world_depth_offset=repeat*128.f;
                assert(render(false,false,false)==ordinary && render(true,false,false)==ordinary);
                assert(render(false,false,true)==coplanar && render(true,false,true)==coplanar);
            }
        }
        std::printf("PASS parameter append/wrap and DISCARD fallback: supported=%u uploads=%u discards=%u\n",
            unsigned(supported),parameters.uploads,parameters.discards);
        short_parameters=false;
        parameters.clear();assert(parameters.available(device,context));
        world_depth_offset=0;assert(render(true,false,false)==ordinary);
        std::puts("PASS selected parameter stream: exact color/depth, first/middle/last ranges, DISCARD lifetimes and reset");
    }else std::puts("UNAVAILABLE constant-buffer offsets: ordinary binding contract retained");
    packed_mesh=true;
    for(bool narrow:{false,true})for(bool stream:{false,true}){
        narrow_mesh=narrow;streamed=stream && parameters.available(device,context);
        for(float origin:{-4096.f,0.f,4096.f}){
            world_depth_offset=origin;
            assert(render(false,false,false)==ordinary && render(true,false,false)==ordinary);
            assert(render(false,false,true)==coplanar && render(true,false,true)==coplanar);
        }
    }
    {
        c3x_renderer::render_core::ImmutableMeshUpload upload;unsigned value=0;
        try{upload.append(&value,upload.limit+1);assert(false);}catch(std::length_error const&){}
        try{upload.append(nullptr,1);assert(false);}catch(std::length_error const&){}
        assert(upload.size()==0);
    }
    std::puts("PASS immutable content allocation: nonzero vertex/index ranges, both index widths, in-flight release, exact color/depth with parameter streams");
    parameters.clear();context->ClearState();vertices->Release();color_buffer->Release();viewport_buffer->Release();rasterizer->Release();depth_state->Release();readback->Release();resolved->Release();dsv->Release();depth->Release();target->Release();color->Release();layout->Release();ps->Release();vs->Release();context->Release();device->Release();
}
