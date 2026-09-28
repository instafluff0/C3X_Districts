#pragma once
namespace c3x_gpu_images {
// Resample an immutable map overlay into display coordinates, then compose it
// over the already projected scene. Geometry is never reconstructed here.
class ProjectedLayer {
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader;
    Microsoft::WRL::ComPtr<ID3D11Buffer> constants;
public:
    void draw(ID3D11Device* device,ID3D11DeviceContext* context,
            ID3D11ShaderResourceView* source,ID3D11ShaderResourceView* below,
            ID3D11UnorderedAccessView* target,Rect area,float scale,
            float cx,float cy,float source_x,float source_y){
        auto check=[](HRESULT h){if(FAILED(h))throw std::runtime_error("projected layer failed");};
        if(!shader){
            char const* code=R"(
Texture2D<uint> source_image:register(t0),below_image:register(t1);
RWTexture2D<uint> output_image:register(u0);
cbuffer Settings:register(b0){float4 mapping;float4 region;float2 offset;uint blend;uint pad;}
float4 unpack(uint p){return float4(p&255,(p>>8)&255,(p>>16)&255,p>>24);}
float4 pixel(int2 p){uint w,h;source_image.GetDimensions(w,h);
 return unpack(source_image.Load(int3(clamp(p,int2(0,0),int2(w-1,h-1)),0)));}
[numthreads(8,8,1)]void main(uint3 at:SV_DispatchThreadID){
 if(any(at.xy>=uint2(region.zw)))return;
 float2 p=(float2(at.xy)+region.xy+.5-mapping.xy)/mapping.z+mapping.xy-.5+offset;
 int2 lo=int2(floor(p));float2 f=frac(p);
 float4 c=lerp(lerp(pixel(lo),pixel(lo+int2(1,0)),f.x),
               lerp(pixel(lo+int2(0,1)),pixel(lo+int2(1,1)),f.x),f.y);
 if(blend){float4 b=unpack(below_image.Load(int3(at.xy,0)));c=c+b*(1-c.a/255);}
 uint4 q=uint4(round(clamp(c,0,255)));output_image[at.xy]=q.x|(q.y<<8)|(q.z<<16)|(q.w<<24);
})";
            Microsoft::WRL::ComPtr<ID3DBlob> compiled,error;
            check(D3DCompile(code,std::strlen(code),"projected layer",nullptr,nullptr,"main","cs_5_0",0,0,&compiled,&error));
            check(device->CreateComputeShader(compiled->GetBufferPointer(),compiled->GetBufferSize(),nullptr,&shader));
            D3D11_BUFFER_DESC d={};d.ByteWidth=48;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            check(device->CreateBuffer(&d,nullptr,&constants));
        }
        struct Parameters{float mapping[4],region[4],offset[2];unsigned blend,pad;} p={
            {cx,cy,scale,0},{float(area.left),float(area.top),float(area.right-area.left),float(area.bottom-area.top)},
            {source_x,source_y},below?1u:0u,0};
        context->ClearState();context->UpdateSubresource(constants.Get(),0,nullptr,&p,0,0);
        auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
        ID3D11ShaderResourceView* inputs[]={source,below};context->CSSetShaderResources(0,2,inputs);
        context->CSSetUnorderedAccessViews(0,1,&target,nullptr);context->CSSetShader(shader.Get(),nullptr,0);
        context->Dispatch((area.right-area.left+7)/8,(area.bottom-area.top+7)/8,1);context->ClearState();
    }
};
}
