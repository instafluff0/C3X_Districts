#pragma once
// Migration adapter only: records the frozen Lab's resource/submission calls.
// It never creates a graphics device. Both actual backends consume packet_v1.h.
#include "../contracts/packet_v1.h"
#include <cstdarg>
#include <cstring>
using UINT = unsigned;
using HRESULT = long;
using DXGI_FORMAT = unsigned;
using D3D_FEATURE_LEVEL = unsigned;
constexpr unsigned DXGI_FORMAT_UNKNOWN = 0, DXGI_FORMAT_R32G32B32A32_FLOAT = 2,
                   DXGI_FORMAT_R32G32B32_FLOAT = 6,
                   DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
                   DXGI_FORMAT_R16G16B16A16_UNORM = 11,
                   DXGI_FORMAT_R32G32_FLOAT = 16, DXGI_FORMAT_R16G16_UNORM = 35,
                   DXGI_FORMAT_R32_FLOAT = 41, DXGI_FORMAT_R8_UNORM = 61,
                   DXGI_FORMAT_R8_UINT = 62, DXGI_FORMAT_BC1_UNORM = 71,
                   DXGI_FORMAT_BC1_UNORM_SRGB = 72, DXGI_FORMAT_BC3_UNORM = 77,
                   DXGI_FORMAT_BC3_UNORM_SRGB = 78, DXGI_FORMAT_BC4_UNORM = 80,
                   DXGI_FORMAT_BC5_UNORM = 83, DXGI_FORMAT_B8G8R8A8_UNORM = 87,
                   DXGI_FORMAT_D24_UNORM_S8_UINT = 45;
constexpr unsigned D3D11_USAGE_IMMUTABLE = 1, D3D11_USAGE_DEFAULT = 2,
                   D3D11_USAGE_STAGING = 3, D3D11_BIND_VERTEX_BUFFER = 1,
                   D3D11_BIND_CONSTANT_BUFFER = 2,
                   D3D11_BIND_SHADER_RESOURCE = 4, D3D11_BIND_RENDER_TARGET = 8,
                   D3D11_BIND_DEPTH_STENCIL = 16, D3D11_CPU_ACCESS_READ = 1,
                   D3D11_FILTER_ANISOTROPIC = 1, D3D11_TEXTURE_ADDRESS_WRAP = 1,
                   D3D11_TEXTURE_ADDRESS_CLAMP = 2, D3D11_FILL_SOLID = 1,
                   D3D11_CULL_NONE = 1, D3D11_BLEND_SRC_ALPHA = 1,
                   D3D11_BLEND_INV_SRC_ALPHA = 2, D3D11_BLEND_ONE = 3,
                   D3D11_BLEND_OP_ADD = 1, D3D11_COLOR_WRITE_ENABLE_ALL = 15,
                   D3D11_DEPTH_WRITE_MASK_ALL = 1,
                   D3D11_COMPARISON_LESS_EQUAL = 1, D3D11_CLEAR_DEPTH = 1,
                   D3D11_INPUT_PER_VERTEX_DATA = 0,
                   D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST = 1,
                   D3D11_MAP_READ = 1, D3D_FEATURE_LEVEL_11_0 = 0,
                   D3D_DRIVER_TYPE_HARDWARE = 0, D3D_DRIVER_TYPE_WARP = 0,
                   D3D11_SDK_VERSION = 0, TRUE = 1;
constexpr float D3D11_FLOAT32_MAX = 3.402823466e+38F;
inline bool SUCCEEDED(HRESULT h) { return h == 0; }
inline bool FAILED(HRESULT h) { return h != 0; }
inline int fopen_s(FILE **f, const char *p, const char *m) {
  *f = fopen(p, m);
  return *f ? 0 : 1;
}
template <size_t N, class... A>
int sprintf_s(char (&s)[N], const char *f, A... a) {
  return snprintf(s, N, f, a...);
}
struct D3D11_SUBRESOURCE_DATA {
  const void *pSysMem = nullptr;
  UINT SysMemPitch = 0, SysMemSlicePitch = 0;
};
struct D3D11_TEXTURE2D_DESC {
  UINT Width = 0, Height = 0, MipLevels = 0, ArraySize = 0, Format = 0;
  struct {
    UINT Count = 0, Quality = 0;
  } SampleDesc;
  UINT Usage = 0, BindFlags = 0, CPUAccessFlags = 0, MiscFlags = 0;
};
struct D3D11_BUFFER_DESC {
  UINT ByteWidth = 0, Usage = 0, BindFlags = 0, CPUAccessFlags = 0,
       MiscFlags = 0, StructureByteStride = 0;
};
struct D3D11_SAMPLER_DESC {
  UINT Filter = 0, AddressU = 0, AddressV = 0, AddressW = 0, MaxAnisotropy = 0;
  float MaxLOD = 0;
};
struct D3D11_RASTERIZER_DESC {
  UINT FillMode = 0, CullMode = 0, DepthClipEnable = 0;
};
struct D3D11_BLEND_DESC {
  struct {
    UINT BlendEnable = 0, SrcBlend = 0, DestBlend = 0, BlendOp = 0,
         SrcBlendAlpha = 0, DestBlendAlpha = 0, BlendOpAlpha = 0,
         RenderTargetWriteMask = 0;
  } RenderTarget[1];
};
struct D3D11_DEPTH_STENCIL_DESC {
  UINT DepthEnable = 0, DepthWriteMask = 0, DepthFunc = 0;
};
struct D3D11_VIEWPORT {
  float TopLeftX, TopLeftY, Width, Height, MinDepth, MaxDepth;
};
struct D3D11_INPUT_ELEMENT_DESC {
  const char *SemanticName;
  UINT SemanticIndex, Format, InputSlot, AlignedByteOffset, InputSlotClass,
      InstanceDataStepRate;
};
struct D3D11_MAPPED_SUBRESOURCE {
  void *pData = nullptr;
  UINT RowPitch = 0;
};
struct RecordObject {
  UINT id = 0;
  void Release() { delete this; }
};
using ID3D11Buffer = RecordObject;
using ID3D11Texture2D = RecordObject;
using ID3D11ShaderResourceView = RecordObject;
using ID3D11RenderTargetView = RecordObject;
using ID3D11DepthStencilView = RecordObject;
using ID3D11DepthStencilState = RecordObject;
using ID3D11SamplerState = RecordObject;
using ID3D11RasterizerState = RecordObject;
using ID3D11BlendState = RecordObject;
using ID3D11VertexShader = RecordObject;
using ID3D11PixelShader = RecordObject;
struct ID3D11InputLayout : RecordObject {
  std::vector<labv2::Attribute> attributes;
};
struct ID3DBlob : RecordObject {
  std::string name;
  void *GetBufferPointer() { return name.data(); }
  size_t GetBufferSize() { return name.size() + 1; }
};
inline labv2::Packet recorded;
struct ID3D11Device : RecordObject {
  HRESULT CreateTexture2D(D3D11_TEXTURE2D_DESC *d,
                          const D3D11_SUBRESOURCE_DATA *s,
                          ID3D11Texture2D **o) {
    *o = new RecordObject;
    if (!s)
      return 0;
    labv2::Texture t;
    t.width = d->Width;
    t.height = d->Height;
    t.format = d->Format;
    for (UINT i = 0; i < d->MipLevels; i++) {
      labv2::Mip m;
      m.pitch = s[i].SysMemPitch;
      size_t n = s[i].SysMemSlicePitch;
      if (!n)
        n = m.pitch * std::max(1u, d->Height >> i);
      auto b = (const uint8_t *)s[i].pSysMem;
      m.bytes.assign(b, b + n);
      t.mips.push_back(std::move(m));
    }
    recorded.textures.push_back(std::move(t));
    (*o)->id = UINT(recorded.textures.size());
    return 0;
  }
  HRESULT CreateShaderResourceView(RecordObject *t, void *, RecordObject **o) {
    *o = new RecordObject{t->id};
    return 0;
  }
  HRESULT CreateRenderTargetView(RecordObject *, void *, RecordObject **o) {
    *o = new RecordObject;
    return 0;
  }
  HRESULT CreateDepthStencilView(RecordObject *, void *, RecordObject **o) {
    *o = new RecordObject;
    return 0;
  }
  HRESULT CreateBuffer(D3D11_BUFFER_DESC *d, D3D11_SUBRESOURCE_DATA *s,
                       RecordObject **o) {
    *o = new RecordObject{UINT(recorded.buffers.size())};
    auto b = (const uint8_t *)s->pSysMem;
    recorded.buffers.emplace_back(b, b + d->ByteWidth);
    return 0;
  }
  HRESULT CreateVertexShader(void *p, size_t, void *, RecordObject **o) {
    *o = new RecordObject{UINT(strstr((char *)p, "Feature") != nullptr)};
    return 0;
  }
  HRESULT CreatePixelShader(void *p, size_t n, void *x, RecordObject **o) {
    return CreateVertexShader(p, n, x, o);
  }
  HRESULT CreateInputLayout(D3D11_INPUT_ELEMENT_DESC *e, UINT n, void *, size_t,
                            ID3D11InputLayout **o) {
    *o = new ID3D11InputLayout;
    for (UINT i = 0; i < n; i++) {
      UINT c = e[i].Format == 2    ? 4
               : e[i].Format == 6  ? 3
               : e[i].Format == 16 ? 2
                                   : 1;
      (*o)->attributes.push_back({c, e[i].AlignedByteOffset});
    }
    return 0;
  }
  HRESULT CreateSamplerState(D3D11_SAMPLER_DESC *, RecordObject **o) {
    *o = new RecordObject;
    return 0;
  }
  HRESULT CreateRasterizerState(D3D11_RASTERIZER_DESC *, RecordObject **o) {
    *o = new RecordObject;
    return 0;
  }
  HRESULT CreateBlendState(D3D11_BLEND_DESC *, RecordObject **o) {
    *o = new RecordObject;
    return 0;
  }
  HRESULT CreateDepthStencilState(D3D11_DEPTH_STENCIL_DESC *,
                                  RecordObject **o) {
    *o = new RecordObject;
    return 0;
  }
};
struct ID3D11DeviceContext : RecordObject {
  labv2::Draw draw;
  void ClearRenderTargetView(RecordObject *, float *) {}
  void ClearDepthStencilView(RecordObject *, UINT, float, UINT) {
    draw.clear_depth = 1;
  }
  void OMSetRenderTargets(UINT, RecordObject **, RecordObject *) {}
  void OMSetDepthStencilState(RecordObject *p, UINT) {
    draw.depth = p != nullptr;
  }
  void OMSetBlendState(RecordObject *, void *, UINT) {}
  void RSSetViewports(UINT, D3D11_VIEWPORT *) {}
  void RSSetState(RecordObject *) {}
  void IASetVertexBuffers(UINT, UINT, RecordObject **p, UINT *s, UINT *) {
    draw.vertex_buffer = (*p)->id;
    draw.stride = *s;
  }
  void IASetInputLayout(ID3D11InputLayout *p) {
    draw.attributes = p->attributes;
  }
  void IASetPrimitiveTopology(UINT) {}
  void VSSetShader(RecordObject *p, void *, UINT) { draw.feature = p->id; }
  void PSSetShader(RecordObject *, void *, UINT) {}
  void PSSetShaderResources(UINT start, UINT n, RecordObject **p) {
    for (UINT i = 0; i < n; i++)
      draw.textures.at(start + i) = p[i] ? p[i]->id : 0;
  }
  void PSSetSamplers(UINT, UINT, RecordObject **) {}
  void PSSetConstantBuffers(UINT, UINT, RecordObject **p) {
    draw.constant_buffer = (*p)->id;
  }
  void Draw(UINT n, UINT) {
    draw.count = n;
    if (n)
      recorded.draws.push_back(draw);
    draw.clear_depth = 0;
  }
  void CopyResource(RecordObject *, RecordObject *) {}
  HRESULT Map(RecordObject *, UINT, UINT, UINT, D3D11_MAPPED_SUBRESOURCE *) {
    return 0;
  }
  void Unmap(RecordObject *, UINT) {}
};
inline HRESULT D3D11CreateDevice(void *, UINT, void *, UINT, UINT *, UINT, UINT,
                                 ID3D11Device **d, void *,
                                 ID3D11DeviceContext **c) {
  *d = new ID3D11Device;
  *c = new ID3D11DeviceContext;
  return 0;
}
