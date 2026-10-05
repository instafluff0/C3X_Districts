"""Bounded diagnostic selection and real-D3D empty-scissor restoration."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import windows_root
from Renderer.tools.measure_redraw_causal import transformed, API, material_probe

ROOT=Path(__file__).resolve().parents[2]


class RedrawCausal(unittest.TestCase):
    def test_material_probe_targets_the_entry_used_by_main_ground(self):
        runtime=(ROOT/'Renderer/native/source_fidelity/runtime.h').read_text()
        self.assertIn('compile_cached(wide.c_str(),"PSFeature","ps_5_0"',runtime)
        constants='\n'.join(x for x in material_probe.__code__.co_consts if isinstance(x,str))
        self.assertIn('Output PSFeature(P input)',constants)
        self.assertNotIn('Output PSMain(P input)',constants)

    def test_only_geometry_submission_changes(self):
        for name,expected in (('fresh_pipeline.h',8),('direct_units.h',3)):
            relative='Renderer/sandbox/'+name
            before=(ROOT/relative).read_text();after,counts=transformed(relative,before)
            self.assertEqual(sum(counts.values()),expected)
            for source in ('auto clip=source_bounds(', 'chunk_intersects_region(',
                           'context->Clear', 'context->Copy', 'context->UpdateSubresource',
                           'context->Draw(3', 'context->DrawIndexed(part.index_count'):
                self.assertEqual(before.count(source),after.count(source))
            # No CPU bounds or selection branches depend on the diagnostic.
            self.assertNotIn('source_bounds(settings,empty',after)
            self.assertNotIn('source_bounds(viewport,empty',after)
        source=(ROOT/'Renderer/sandbox/direct_units.h').read_text()
        after,_=transformed('Renderer/sandbox/direct_units.h',source)
        shadow=after.split('bool draw_self_shadow(',1)[1].split('bool draw_real(',1)[0]
        self.assertNotIn('sandbox_causal.issue',shadow)
        self.assertNotIn('SandboxCausalRaster',shadow)

    def test_ledger_keeps_arguments_and_counts_suppressed_draws_correctly(self):
        text='if(sandbox_causal.issue())context->DrawIndexed(7,0,0);renderer.context->CopyResource(a,b);'
        result=API.sub(lambda m:'(sandbox_causal.api("'+m[2]+'"),'+m[1]+')->'+m[2],text)
        self.assertEqual(result,'if(sandbox_causal.issue())(sandbox_causal.api("DrawIndexed"),context)->DrawIndexed(7,0,0);(sandbox_causal.api("CopyResource"),renderer.context)->CopyResource(a,b);')

    def test_empty_raster_preserves_color_depth_and_restores_hardware_state(self):
        include=str(windows_root()/'Renderer/sandbox/redraw_causal.h')
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <vector>
#include <cassert>
#include "''' + include + r'''"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
std::vector<unsigned char> read(ID3D11Device*d,ID3D11DeviceContext*c,ID3D11Texture2D*t){
 D3D11_TEXTURE2D_DESC td={};t->GetDesc(&td);td.BindFlags=td.MiscFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Texture2D>s;assert(SUCCEEDED(d->CreateTexture2D(&td,nullptr,&s)));c->CopyResource(s.Get(),t);
 D3D11_MAPPED_SUBRESOURCE m={};assert(SUCCEEDED(c->Map(s.Get(),0,D3D11_MAP_READ,0,&m)));std::vector<unsigned char>v(td.Width*td.Height*4);
 for(unsigned y=0;y<td.Height;++y)std::memcpy(v.data()+y*td.Width*4,static_cast<char*>(m.pData)+y*m.RowPitch,td.Width*4);c->Unmap(s.Get(),0);return v;
}
int main(){ComPtr<ID3D11Device>d;ComPtr<ID3D11DeviceContext>c;D3D_FEATURE_LEVEL fl;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&d,&fl,&c)));
 const char*src=R"(float4 VS(uint i:SV_VertexID):SV_POSITION{return float4(i==1?3:-1,i==2?-3:1,.4,1);}float4 PS():SV_Target{return float4(1,.2,.3,1);})";
 ComPtr<ID3DBlob>v,p;assert(SUCCEEDED(D3DCompile(src,std::strlen(src),nullptr,nullptr,nullptr,"VS","vs_5_0",0,0,&v,nullptr)));
 assert(SUCCEEDED(D3DCompile(src,std::strlen(src),nullptr,nullptr,nullptr,"PS","ps_5_0",0,0,&p,nullptr)));
 ComPtr<ID3D11VertexShader>vs;ComPtr<ID3D11PixelShader>ps;assert(SUCCEEDED(d->CreateVertexShader(v->GetBufferPointer(),v->GetBufferSize(),nullptr,&vs)));
 assert(SUCCEEDED(d->CreatePixelShader(p->GetBufferPointer(),p->GetBufferSize(),nullptr,&ps)));
 D3D11_TEXTURE2D_DESC td={};td.Width=td.Height=16;td.ArraySize=td.MipLevels=td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;td.BindFlags=D3D11_BIND_RENDER_TARGET;
 ComPtr<ID3D11Texture2D>t,z;ComPtr<ID3D11RenderTargetView>rt;ComPtr<ID3D11DepthStencilView>dv;
 assert(SUCCEEDED(d->CreateTexture2D(&td,nullptr,&t)));assert(SUCCEEDED(d->CreateRenderTargetView(t.Get(),nullptr,&rt)));auto target=rt.Get();
 td.Format=DXGI_FORMAT_R24G8_TYPELESS;td.BindFlags=D3D11_BIND_DEPTH_STENCIL;assert(SUCCEEDED(d->CreateTexture2D(&td,nullptr,&z)));
 D3D11_DEPTH_STENCIL_VIEW_DESC dd={};dd.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;dd.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
 assert(SUCCEEDED(d->CreateDepthStencilView(z.Get(),&dd,&dv)));c->OMSetRenderTargets(1,&target,dv.Get());
 D3D11_DEPTH_STENCIL_DESC ds={};ds.DepthEnable=TRUE;ds.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;ds.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
 ComPtr<ID3D11DepthStencilState>depth;assert(SUCCEEDED(d->CreateDepthStencilState(&ds,&depth)));c->OMSetDepthStencilState(depth.Get(),0);
 D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;rd.ScissorEnable=TRUE;rd.DepthClipEnable=TRUE;
 ComPtr<ID3D11RasterizerState>rs;assert(SUCCEEDED(d->CreateRasterizerState(&rd,&rs)));c->RSSetState(rs.Get());
 D3D11_VIEWPORT vp={0,0,16,16,0,1};c->RSSetViewports(1,&vp);D3D11_RECT original[2]={{2,3,14,15},{1,1,8,8}};c->RSSetScissorRects(2,original);
 c->VSSetShader(vs.Get(),nullptr,0);c->PSSetShader(ps.Get(),nullptr,0);c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
 unsigned indices[]={0,1,2};D3D11_BUFFER_DESC bd={};bd.ByteWidth=sizeof(indices);bd.BindFlags=D3D11_BIND_INDEX_BUFFER;
 D3D11_SUBRESOURCE_DATA data={indices,0,0};ComPtr<ID3D11Buffer>ib;assert(SUCCEEDED(d->CreateBuffer(&bd,&data,&ib)));c->IASetIndexBuffer(ib.Get(),DXGI_FORMAT_R32_UINT,0);
 unsigned pass=2;SetEnvironmentVariableA("C3X_SANDBOX_CAUSAL_LEDGER","1");
 for(unsigned mode=0;mode<3;++mode){char value[2]={char('0'+mode),0};SetEnvironmentVariableA("C3X_SANDBOX_CAUSAL_MODE",value);
  float clear[]={0,0,0,0};c->ClearRenderTargetView(target,clear);c->ClearDepthStencilView(dv.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,5);
  auto before=read(d.Get(),c.Get(),t.Get()),before_depth=read(d.Get(),c.Get(),z.Get());
  {SandboxCausalFrame frame(&pass,29533);{SandboxCausalRaster scope(c.Get());
    if(sandbox_causal.issue())c->DrawIndexed(3,0,0);{SandboxCausalRaster nested(c.Get());if(sandbox_causal.issue())c->DrawIndexedInstanced(3,2,0,0,0);}
    if(sandbox_causal.issue())c->DrawIndexed(3,0,0);}
   assert(sandbox_causal.depth==0&&sandbox_causal.bad_state==0);assert(sandbox_causal.attempts==3);assert(sandbox_causal.suppressed==(mode==2?3u:0u));
   D3D11_RECT actual[16]={};UINT n=16;c->RSGetScissorRects(&n,actual);assert(n==2&&!std::memcmp(actual,original,sizeof(original)));
   ComPtr<ID3D11RasterizerState>actual_rs;c->RSGetState(&actual_rs);assert(actual_rs.Get()==rs.Get());
   auto color=read(d.Get(),c.Get(),t.Get()),depth_bytes=read(d.Get(),c.Get(),z.Get());
   assert((color==before)==(mode!=0));assert((depth_bytes==before_depth)==(mode!=0));
   // Shadow preparation/fullscreen draws outside the geometry scope still run.
   c->Draw(3,0);assert(read(d.Get(),c.Get(),t.Get())!=before);
  }
 }
 std::puts("CAUSAL_HARDWARE pass indexed=1 instanced=1 empty_color_depth_exact=1 nested_restore=1 outside_draw=1");
}
''',timeout=60)


if __name__=='__main__':unittest.main()
