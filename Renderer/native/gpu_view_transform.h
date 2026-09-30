#pragma once
namespace c3x_gpu_images {
// A resolved world image is transformed before fixed interface composition.
// Packed BGRA is filtered explicitly; integer textures cannot use a sampler.
class ViewTransform {
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader;
    Microsoft::WRL::ComPtr<ID3D11Buffer> constants;
public:
    void draw(ID3D11Device* device,ID3D11DeviceContext* context,
              ID3D11ShaderResourceView* source,ID3D11UnorderedAccessView* destination,
              unsigned width,unsigned height,float scale,
              ID3D11UnorderedAccessView* native_words=nullptr,unsigned native_format=0){
        auto check=[](HRESULT hr){if(FAILED(hr))throw std::runtime_error("world view transform failed");};
        if(!shader){
            char const* code=R"(
Texture2D<uint> source_image:register(t0);
RWTexture2D<uint> destination_image:register(u0);
RWTexture2D<uint> native_image:register(u1);
cbuffer View:register(b0){uint width,height;float scale;uint native_format;}
float4 unpack(uint p){return float4(p&255,(p>>8)&255,(p>>16)&255,p>>24);}
float4 pixel(int2 p){return unpack(source_image.Load(int3(clamp(p,int2(0,0),int2(width-1,height-1)),0)));}
[numthreads(8,8,1)] void main(uint3 at:SV_DispatchThreadID){
 if(at.x>=width||at.y>=height)return;
 uint4 c;
 // Geometry-projected scenes already have display-resolution pixels. Preserve
 // their exact bytes with one fetch while producing the native color pair.
 [branch] if(scale==1) c=uint4(unpack(source_image.Load(int3(at.xy,0))));
 else {
  float2 center=float2(width/2,height/2);
  float2 p=(float2(at.xy)+.5-center)/scale+center-.5;
  int2 lo=int2(floor(p));float2 f=frac(p);
  c=uint4(round(lerp(lerp(pixel(lo),pixel(lo+int2(1,0)),f.x),
                     lerp(pixel(lo+int2(0,1)),pixel(lo+int2(1,1)),f.x),f.y)));
 }
 destination_image[at.xy]=c.x|(c.y<<8)|(c.z<<16)|(c.w<<24);
 if(native_format){
  uint threshold=0;
  [unroll] for(uint bit=0;bit<3;++bit){
   uint a=(at.x>>bit)&1,b=(at.y>>bit)&1;
   threshold=(threshold<<2)|((a^b)<<1)|b;
  }
  uint3 levels=uint3(31,native_format==2?63:31,31),scaled=c.xyz*levels;
  uint3 q=scaled/255+uint3((scaled%255)*128>(threshold*2+1)*255);
  native_image[at.xy]=q.x|(q.y<<5)|(q.z<<(native_format==2?11:10));
 }
}
)";
            Microsoft::WRL::ComPtr<ID3DBlob> compiled,error;
            check(D3DCompile(code,std::strlen(code),"retained world zoom",nullptr,nullptr,"main","cs_5_0",0,0,&compiled,&error));
            check(device->CreateComputeShader(compiled->GetBufferPointer(),compiled->GetBufferSize(),nullptr,&shader));
            D3D11_BUFFER_DESC desc={};desc.ByteWidth=16;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            check(device->CreateBuffer(&desc,nullptr,&constants));
        }
        struct Parameters {unsigned width,height;float scale;unsigned native_format;} parameters={width,height,scale,native_format};
        context->ClearState();
        context->UpdateSubresource(constants.Get(),0,nullptr,&parameters,0,0);
        auto buffer=constants.Get();context->CSSetConstantBuffers(0,1,&buffer);
        ID3D11UnorderedAccessView* outputs[]={destination,native_words};
        context->CSSetShaderResources(0,1,&source);context->CSSetUnorderedAccessViews(0,2,outputs,nullptr);
        context->CSSetShader(shader.Get(),nullptr,0);context->Dispatch((width+7)/8,(height+7)/8,1);
        context->ClearState();
    }
};
}
