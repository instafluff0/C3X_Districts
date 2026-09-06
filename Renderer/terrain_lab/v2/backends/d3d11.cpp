#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "../contracts/packet_v1.h"
#include "../shared/color_response.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <fstream>
#include <windows.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
using namespace labv2;
void check(HRESULT h, const char *msg) {
  if (FAILED(h))
    throw std::runtime_error(std::string(msg) +
                             " HRESULT=" + std::to_string(h));
}
ComPtr<ID3DBlob> compile(const char *path, const char *entry,
                         const char *target) {
  FILE* f=open_path(path,"rb");
  if(!f)throw std::runtime_error("missing HLSL module");
  std::string source;char block[8192];size_t n;while((n=fread(block,1,sizeof(block),f)))source.append(block,n);fclose(f);
  if (source.empty())
    throw std::runtime_error("missing HLSL module");
  ComPtr<ID3DBlob> b, e;
  HRESULT hr =
      D3DCompile(source.data(), source.size(), nullptr, nullptr, nullptr, entry,
                 target, D3DCOMPILE_ENABLE_STRICTNESS, 0, &b, &e);
  if (FAILED(hr) && e)
    throw std::runtime_error(
        std::string((char *)e->GetBufferPointer(), e->GetBufferSize()));
  check(hr, "shader compile");
  return b;
}
int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 9)
      throw std::runtime_error("usage: d3d11 packet shader.hlsl output.bmp");
    Packet p = read_packet(argv[1]);
    bool linear = p.color_branch == 1;
    unsigned samples=argc>4?std::stoul(argv[4]):1;
    std::string post=argc>5?argv[5]:"box";
    float offset_x=argc>6?std::stof(argv[6]):0,offset_y=argc>7?std::stof(argv[7]):0;
    unsigned scale=argc>8?std::stoul(argv[8]):1;
    if((scale!=1&&scale!=2&&scale!=4)||p.width*scale>8192||p.height*scale>8192)throw std::runtime_error("unsupported D3D render scale");
    if(post!="box"&&!linear)throw std::runtime_error("D3D custom post requires linear contract2");
    unsigned w=p.width*scale,h=p.height*scale;
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                            &level, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx),
          "device");
    std::vector<ComPtr<ID3D11ShaderResourceView>> textures(1);
    for (auto &t : p.textures) {
      D3D11_TEXTURE2D_DESC d = {};
      d.Width = t.width;
      d.Height = t.height;
      d.MipLevels = UINT(t.mips.size());
      d.ArraySize = 1;
      d.Format = DXGI_FORMAT(t.format);
      d.SampleDesc.Count = 1;
      d.Usage = D3D11_USAGE_IMMUTABLE;
      d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      std::vector<D3D11_SUBRESOURCE_DATA> data;
      for (auto &m : t.mips)
        data.push_back({m.bytes.data(), m.pitch, UINT(m.bytes.size())});
      ComPtr<ID3D11Texture2D> texture;
      check(dev->CreateTexture2D(&d, data.data(), &texture), "texture");
      ComPtr<ID3D11ShaderResourceView> view;
      check(dev->CreateShaderResourceView(texture.Get(), nullptr, &view),
            "texture view");
      textures.push_back(view);
    }
    std::vector<ComPtr<ID3D11Buffer>> buffers;
    for (size_t i = 0; i < p.buffers.size(); i++) {
      bool constant = false;
      for (auto &d : p.draws)
        if (d.constant_buffer == i || d.frame_buffer == i)
          constant = true;
      D3D11_BUFFER_DESC desc = {};
      desc.ByteWidth = UINT(p.buffers[i].size());
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags =
          constant ? D3D11_BIND_CONSTANT_BUFFER : D3D11_BIND_VERTEX_BUFFER;
      D3D11_SUBRESOURCE_DATA data = {p.buffers[i].data(), 0, 0};
      ComPtr<ID3D11Buffer> b;
      check(dev->CreateBuffer(&desc, &data, &b), "buffer");
      buffers.push_back(b);
    }
    std::vector<ComPtr<ID3DBlob>> blobs(p.shader_count*2);
    std::vector<ComPtr<ID3D11VertexShader>> vs(p.shader_count*2);
    std::vector<ComPtr<ID3D11PixelShader>> ps(p.shader_count*2);
    for (unsigned i = 0; i < p.shader_count*2; i++) {
      std::string source=argv[2];
      if(p.shader_count>1) source+="/m"+std::to_string(i/2)+".hlsl";
      blobs[i] = compile(source.c_str(), i%2 ? "VSFeature" : "VSMain", "vs_5_0");
      auto pixel = compile(source.c_str(), i%2 ? "PSFeature" : "PSMain", "ps_5_0");
      check(dev->CreateVertexShader(blobs[i]->GetBufferPointer(),
                                    blobs[i]->GetBufferSize(), nullptr, &vs[i]),
            "vertex shader");
      check(dev->CreatePixelShader(pixel->GetBufferPointer(),
                                   pixel->GetBufferSize(), nullptr, &ps[i]),
            "pixel shader");
    }
    ComPtr<ID3D11SamplerState> samplers[2];
    for (unsigned i = 0; i < 2; i++) {
      D3D11_SAMPLER_DESC d = {};
      d.Filter = D3D11_FILTER_ANISOTROPIC;
      d.AddressU = d.AddressV = d.AddressW =
          i ? D3D11_TEXTURE_ADDRESS_CLAMP : D3D11_TEXTURE_ADDRESS_WRAP;
      d.MaxAnisotropy = 8;
      d.MaxLOD = D3D11_FLOAT32_MAX;
      check(dev->CreateSamplerState(&d, &samplers[i]), "sampler");
    }
    ID3D11SamplerState *sp[] = {samplers[0].Get(), samplers[1].Get()};
    ctx->PSSetSamplers(0, 2, sp);
    DXGI_FORMAT format=linear?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_B8G8R8A8_UNORM;
    for(auto f:{format,DXGI_FORMAT_D24_UNORM_S8_UINT,DXGI_FORMAT_R8_UNORM}) {
      UINT levels=0;check(dev->CheckMultisampleQualityLevels(f,samples,&levels),"sample capability");
      if(!levels)throw std::runtime_error("unsupported D3D sample count");
    }
    auto make_texture=[&](DXGI_FORMAT format,unsigned width,unsigned height,unsigned count,UINT bind,bool staging){
      D3D11_TEXTURE2D_DESC td={};td.Width=width;td.Height=height;td.MipLevels=td.ArraySize=1;td.SampleDesc.Count=count;td.Format=format;
      td.BindFlags=bind;td.Usage=staging?D3D11_USAGE_STAGING:D3D11_USAGE_DEFAULT;td.CPUAccessFlags=staging?D3D11_CPU_ACCESS_READ:0;
      ComPtr<ID3D11Texture2D> t;check(dev->CreateTexture2D(&td,nullptr,&t),"attachment allocation");return t;
    };
    auto color=make_texture(format,w,h,samples,D3D11_BIND_RENDER_TARGET|(samples==1?D3D11_BIND_SHADER_RESOURCE:0),false);
    auto resolved=samples==1?color:make_texture(format,w,h,1,D3D11_BIND_SHADER_RESOURCE,false);
    auto readback=make_texture(format,w,h,1,0,true);
    auto depth=make_texture(DXGI_FORMAT_D24_UNORM_S8_UINT,w,h,samples,D3D11_BIND_DEPTH_STENCIL,false);
    ComPtr<ID3D11RenderTargetView> rt;check(dev->CreateRenderTargetView(color.Get(),nullptr,&rt),"target");
    ComPtr<ID3D11DepthStencilView> depthView;check(dev->CreateDepthStencilView(depth.Get(),nullptr,&depthView),"depth view");
    ComPtr<ID3D11Texture2D> validity,valid_resolved,validity_readback;
    ComPtr<ID3D11RenderTargetView> validity_rt;
    if(linear){
      validity=make_texture(DXGI_FORMAT_R8_UNORM,w,h,samples,D3D11_BIND_RENDER_TARGET|(samples==1?D3D11_BIND_SHADER_RESOURCE:0),false);
      valid_resolved=samples==1?validity:make_texture(DXGI_FORMAT_R8_UNORM,w,h,1,D3D11_BIND_SHADER_RESOURCE,false);
      validity_readback=make_texture(DXGI_FORMAT_R8_UNORM,w,h,1,0,true);
      check(dev->CreateRenderTargetView(validity.Get(),nullptr,&validity_rt),"validity target");
    }
    D3D11_DEPTH_STENCIL_DESC ds = {};
    ds.DepthEnable = TRUE;
    ds.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    ds.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    ComPtr<ID3D11DepthStencilState> depthState;
    check(dev->CreateDepthStencilState(&ds, &depthState), "depth state");
    ComPtr<ID3D11DepthStencilState> depthRead, depthOff;
    ds.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    check(dev->CreateDepthStencilState(&ds,&depthRead),"depth read");
    ds.DepthEnable = FALSE;
    check(dev->CreateDepthStencilState(&ds,&depthOff),"depth off");
    D3D11_RASTERIZER_DESC rs = {};
    rs.FillMode = D3D11_FILL_SOLID;
    rs.CullMode = D3D11_CULL_NONE;
    rs.DepthClipEnable = TRUE;
    rs.MultisampleEnable=samples>1;
    ComPtr<ID3D11RasterizerState> raster;
    check(dev->CreateRasterizerState(&rs, &raster), "raster");
    ctx->RSSetState(raster.Get());
    D3D11_BLEND_DESC bd = {};
    auto &b = bd.RenderTarget[0];
    b.BlendEnable = TRUE;
    b.SrcBlend = D3D11_BLEND_SRC_ALPHA;
    b.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    b.BlendOp = D3D11_BLEND_OP_ADD;
    b.SrcBlendAlpha = D3D11_BLEND_ONE;
    b.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    b.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    b.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    ComPtr<ID3D11BlendState> blend;
    check(dev->CreateBlendState(&bd, &blend), "blend");
    ComPtr<ID3D11BlendState> linearBlends[3];
    if (linear) for (unsigned mode=0;mode<3;mode++) {
      bd.IndependentBlendEnable = TRUE; b.BlendEnable = mode!=0;
      b.SrcBlend = D3D11_BLEND_ONE; b.DestBlend = mode==2?D3D11_BLEND_ONE:D3D11_BLEND_INV_SRC_ALPHA;
      b.RenderTargetWriteMask = mode==2 ? 7 : D3D11_COLOR_WRITE_ENABLE_ALL;
      auto &v=bd.RenderTarget[1]; v=b; v.BlendEnable=TRUE; v.BlendOp=D3D11_BLEND_OP_MAX;
      v.SrcBlend=v.DestBlend=D3D11_BLEND_ONE; v.RenderTargetWriteMask=mode!=0?0:D3D11_COLOR_WRITE_ENABLE_RED;
      check(dev->CreateBlendState(&bd,&linearBlends[mode]),"linear blend");
    }
    ctx->OMSetBlendState(blend.Get(), nullptr, 0xffffffffu);
    float clear[] = {.035f, .035f, .035f, 1};
    if (linear) for (auto &v:clear) v=0;
    ctx->ClearRenderTargetView(rt.Get(), clear);
    if (linear) ctx->ClearRenderTargetView(validity_rt.Get(), clear);
    ctx->ClearDepthStencilView(depthView.Get(),D3D11_CLEAR_DEPTH,1,0);
    D3D11_VIEWPORT vp = {offset_x*scale*p.downsample, offset_y*scale*p.downsample, float(w), float(h), 0, 1};
    ctx->RSSetViewports(1, &vp);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    for (auto &d : p.draws) {
      if (d.clear_depth)
        ctx->ClearDepthStencilView(depthView.Get(), D3D11_CLEAR_DEPTH, 1, 0);
      ID3D11RenderTargetView* targets[]={rt.Get(),validity_rt.Get()};
      ctx->OMSetRenderTargets(linear?2:1, targets, depthView.Get());
      if (linear) {
        ctx->OMSetDepthStencilState(d.depth_mode==2?depthState.Get():d.depth_mode==1?depthRead.Get():depthOff.Get(),0);
        ctx->OMSetBlendState(linearBlends[d.blend_mode].Get(),nullptr,0xffffffffu);
      } else ctx->OMSetDepthStencilState(d.depth ? depthState.Get() : nullptr, 0);
      UINT offset = 0;
      auto vertex = buffers[d.vertex_buffer].Get();
      ctx->IASetVertexBuffers(0, 1, &vertex, &d.stride, &offset);
      auto constant = buffers[d.constant_buffer].Get();
      ctx->PSSetConstantBuffers(0, 1, &constant);
      auto frame=d.frame_buffer==UINT32_MAX?nullptr:buffers[d.frame_buffer].Get();
      ctx->PSSetConstantBuffers(1,1,&frame);
      ctx->VSSetShader(vs[d.shader_index*2+d.feature].Get(), nullptr, 0);
      ctx->PSSetShader(ps[d.shader_index*2+d.feature].Get(), nullptr, 0);
      std::vector<D3D11_INPUT_ELEMENT_DESC> attrs;
      unsigned texcoord = 0;
      for (size_t i = 0; i < d.attributes.size(); i++) {
        bool normal = i == (d.feature ? 2u : 3u);
        const char *semantic = i == 0   ? "POSITION"
                               : normal ? "NORMAL"
                                        : "TEXCOORD";
        unsigned index = i == 0 || normal ? 0 : texcoord++;
        DXGI_FORMAT formats[] = {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32_FLOAT,
                                 DXGI_FORMAT_R32G32_FLOAT,
                                 DXGI_FORMAT_R32G32B32_FLOAT,
                                 DXGI_FORMAT_R32G32B32A32_FLOAT};
        attrs.push_back({semantic, index, formats[d.attributes[i].components],
                         0, d.attributes[i].offset, D3D11_INPUT_PER_VERTEX_DATA,
                         0});
      }
      ComPtr<ID3D11InputLayout> layout;
      check(dev->CreateInputLayout(attrs.data(), UINT(attrs.size()),
                                   blobs[d.shader_index*2+d.feature]->GetBufferPointer(),
                                   blobs[d.shader_index*2+d.feature]->GetBufferSize(), &layout),
            "layout");
      ctx->IASetInputLayout(layout.Get());
      ID3D11ShaderResourceView *views[128] = {};
      for (unsigned i = 0; i < 128; i++)
        views[i] = textures[d.textures[i]].Get();
      ctx->PSSetShaderResources(0, 128, views);
      ctx->Draw(d.count, 0);
    }
    ctx->OMSetRenderTargets(0,nullptr,nullptr);
    if(samples>1){ctx->ResolveSubresource(resolved.Get(),0,color.Get(),0,format);if(linear)ctx->ResolveSubresource(valid_resolved.Get(),0,validity.Get(),0,DXGI_FORMAT_R8_UNORM);}
    unsigned read_w=w,read_h=h,read_scale=scale*p.downsample;
    if(post!="box"){
      read_w=p.width/p.downsample;read_h=p.height/p.downsample;read_scale=1;
      auto post_color=make_texture(format,read_w,read_h,1,D3D11_BIND_UNORDERED_ACCESS,false);
      auto post_valid=make_texture(DXGI_FORMAT_R8_UNORM,read_w,read_h,1,D3D11_BIND_UNORDERED_ACCESS,false);
      ComPtr<ID3D11ShaderResourceView> src,mask;check(dev->CreateShaderResourceView(resolved.Get(),nullptr,&src),"post source");check(dev->CreateShaderResourceView(valid_resolved.Get(),nullptr,&mask),"post validity");
      ComPtr<ID3D11UnorderedAccessView> dst,dst_mask;check(dev->CreateUnorderedAccessView(post_color.Get(),nullptr,&dst),"post output");check(dev->CreateUnorderedAccessView(post_valid.Get(),nullptr,&dst_mask),"post output validity");
      auto blob=compile(post.c_str(),"CSPost","cs_5_0");ComPtr<ID3D11ComputeShader> cs;check(dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs),"post compute");
      uint32_t sizes[]={w,h,read_w,read_h,p.valid_rect[0],p.valid_rect[1],p.valid_rect[2],p.valid_rect[3]};D3D11_BUFFER_DESC bd={};bd.ByteWidth=32;bd.Usage=D3D11_USAGE_IMMUTABLE;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA initial={sizes,0,0};ComPtr<ID3D11Buffer> cb;check(dev->CreateBuffer(&bd,&initial,&cb),"post sizes");
      ID3D11ShaderResourceView* srvs[]={src.Get(),nullptr,nullptr,mask.Get()};ID3D11UnorderedAccessView* uavs[]={nullptr,dst.Get(),nullptr,nullptr,dst_mask.Get()};auto constants=cb.Get();
      ctx->CSSetShader(cs.Get(),nullptr,0);ctx->CSSetShaderResources(0,4,srvs);ctx->CSSetUnorderedAccessViews(0,5,uavs,nullptr);ctx->CSSetConstantBuffers(2,1,&constants);ctx->Dispatch((read_w+7)/8,(read_h+7)/8,1);
      ID3D11UnorderedAccessView* nulls[5]={};ID3D11ShaderResourceView* empty[4]={};ctx->CSSetUnorderedAccessViews(0,5,nulls,nullptr);ctx->CSSetShaderResources(0,4,empty);
      resolved=post_color;valid_resolved=post_valid;readback=make_texture(format,read_w,read_h,1,0,true);validity_readback=make_texture(DXGI_FORMAT_R8_UNORM,read_w,read_h,1,0,true);
    }
    ctx->CopyResource(readback.Get(), resolved.Get());
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    check(ctx->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped), "map");
    std::vector<uint16_t> rgba;
    if (linear) {
      rgba.resize(size_t(read_w)*read_h*4);
      for(unsigned y=0;y<read_h;y++) memcpy(rgba.data()+size_t(y)*read_w*4,(uint8_t*)mapped.pData+size_t(y)*mapped.RowPitch,read_w*8);
      ctx->Unmap(readback.Get(),0);
      ctx->CopyResource(validity_readback.Get(),valid_resolved.Get());
      check(ctx->Map(validity_readback.Get(),0,D3D11_MAP_READ,0,&mapped),"validity map");
      std::vector<uint8_t> mask(size_t(read_w)*read_h);
      for(unsigned y=0;y<read_h;y++) memcpy(mask.data()+size_t(y)*read_w,(uint8_t*)mapped.pData+size_t(y)*mapped.RowPitch,read_w);
      ctx->Unmap(validity_readback.Get(),0);
      auto result=display_pixels(rgba,mask,read_w,read_h,read_scale,p);
      write_attachment(std::string(argv[3])+".linear.rgba16f",rgba.data(),rgba.size()*2);
      write_attachment(std::string(argv[3])+".validity.r8",mask.data(),mask.size());
      if(!write_bmp(argv[3],result,read_w/read_scale,read_h/read_scale,1)) throw std::runtime_error("linear output write failed");
      return 0;
    }
    std::vector<uint8_t> pixels(size_t(read_w) * read_h * 4);
    for (unsigned y = 0; y < read_h; y++)
      memcpy(pixels.data() + size_t(y) * read_w * 4,
             (uint8_t *)mapped.pData + size_t(y) * mapped.RowPitch,
             read_w * 4);
    ctx->Unmap(readback.Get(), 0);
    if (!write_bmp(argv[3], pixels, read_w, read_h, read_scale))
      throw std::runtime_error("output write failed");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "lab-v2 D3D11: %s\n", e.what());
    return 1;
  }
}
