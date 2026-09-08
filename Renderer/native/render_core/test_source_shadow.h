// Headless hardware checks for real source rasterization and page dependencies.
#include "source_shadow.h"
inline bool test_source_shadow(ID3D11Device* device,ID3D11DeviceContext* context) {
 using Shadow=c3x_renderer::render_core::SourceShadow;
 Shadow shadow;if(!shadow.ensure(device,L"source_caster.hlsl"))return false;
 struct V {float prefix[30];float world[4];float remaining[8];};
 V vertices[4]={};float xy[4][2]={{1,1},{3,1},{3,3},{1,3}};
 for(unsigned i=0;i<4;i++){
  vertices[i].prefix[3]=(xy[i][0]-1)*.499f;
  vertices[i].prefix[4]=(xy[i][1]-1)*.499f;
  vertices[i].world[0]=xy[i][0];vertices[i].world[1]=xy[i][1];
  vertices[i].world[2]=vertices[i].world[3]=1;
 }
 unsigned indices[]={0,1,2,0,2,3};
 ID3D11Buffer *vb=nullptr,*ib=nullptr;
 D3D11_BUFFER_DESC b={};b.ByteWidth=sizeof(vertices);b.Usage=D3D11_USAGE_IMMUTABLE;b.BindFlags=D3D11_BIND_VERTEX_BUFFER;
 D3D11_SUBRESOURCE_DATA data={};data.pSysMem=vertices;
 if(FAILED(device->CreateBuffer(&b,&data,&vb)))return false;
 b.ByteWidth=sizeof(indices);b.BindFlags=D3D11_BIND_INDEX_BUFFER;data.pSysMem=indices;
 if(FAILED(device->CreateBuffer(&b,&data,&ib))){vb->Release();return false;}
 Shadow::Caster caster;caster.vertices=vb;caster.indices=ib;caster.count=6;caster.stride=sizeof(V);
 caster.version=1;caster.bounds={{1,1,1},{3,3,1}};
 std::vector<Shadow::Bounds> receivers={{{1,1,0},{3,3,0}}};
 std::array<float,12> basis={1,0,0,6,0,1,0,1024,0,0,1,0};
 auto opaque=[](unsigned){return false;};
 bool ok=shadow.prepare(context,basis,receivers,{caster},opaque,nullptr) && shadow.rebuilt==1 && shadow.draws==1;
 ok=ok && shadow.prepare(context,basis,receivers,{caster},opaque,nullptr) && shadow.hits==1 && shadow.rebuilt==0;
 auto distant=caster;distant.offset[0]=30;distant.version=7;
 ok=ok && shadow.prepare(context,basis,receivers,{caster,distant},opaque,nullptr) && shadow.hits==1 && shadow.rebuilt==0;
 caster.version=2;
 ok=ok && shadow.prepare(context,basis,receivers,{caster,distant},opaque,nullptr) && shadow.rebuilt==1;
 // Sample a hole and solid texel from actual nearest-mip-zero alpha coverage.
 ID3D11Texture2D *alpha=nullptr,*readback=nullptr;ID3D11ShaderResourceView* alpha_view=nullptr;
 unsigned pixels[]={0x00ffffff,0xffffffff,0xffffffff,0xffffffff};
 D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=2;d.MipLevels=d.ArraySize=1;d.SampleDesc.Count=1;
 d.Format=DXGI_FORMAT_R8G8B8A8_UNORM;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 data.pSysMem=pixels;data.SysMemPitch=8;
 if(FAILED(device->CreateTexture2D(&d,&data,&alpha)))ok=false;
 if(ok && FAILED(device->CreateShaderResourceView(alpha,nullptr,&alpha_view)))ok=false;
 caster.version=3;
 auto cutout=[&](unsigned){context->PSSetShaderResources(0,1,&alpha_view);return true;};
 if(ok)ok=shadow.prepare(context,basis,receivers,{caster},cutout,nullptr) && shadow.rebuilt==1;
 d.Width=d.Height=1024;d.Format=DXGI_FORMAT_R32_FLOAT;d.Usage=D3D11_USAGE_STAGING;
 d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 if(ok && FAILED(device->CreateTexture2D(&d,nullptr,&readback)))ok=false;
 if(ok){ID3D11Resource* source=nullptr;shadow.view->GetResource(&source);
  context->CopySubresourceRegion(readback,0,0,0,0,source,0,nullptr);source->Release();
  D3D11_MAPPED_SUBRESOURCE map={};
  if(FAILED(context->Map(readback,0,D3D11_MAP_READ,0,&map)))ok=false;
  else {
   auto sample=[&](float x,float y){auto row=reinterpret_cast<char*>(map.pData)+int(y/6*1024)*map.RowPitch;
    return reinterpret_cast<float*>(row)[int(x/6*1024)];};
   ok=sample(1.5f,1.5f)<-999.f && std::abs(sample(2.5f,2.5f)-1.f)<1e-5f;
   context->Unmap(readback,0);
  }
 }
 context->ClearState();if(readback)readback->Release();if(alpha_view)alpha_view->Release();if(alpha)alpha->Release();
 vb->Release();ib->Release();
 std::printf("render-core source shadows: %s (warm pages, distant edits, source edit, cutout hole)\n",ok?"pass":"FAIL");
 return ok;
}
