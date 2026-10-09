"""The production shadow page field is created on first use, not at start.

Renderer64's resident scene samples its own shadow atlas and never draws the
production field (32 slices of 1024 x 1024 R32F, 128 MiB), which every session
still allocated at device start (performance review, section 20). Set-up now
compiles only shaders and buffers; the first prepare() creates the field.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class SourceShadowLazyFieldTests(unittest.TestCase):
    def test_setup_allocates_no_field_and_prepare_creates_it_once(self):
        source = (ROOT / "Renderer/native/render_core/source_shadow.h").read_text()
        setup = method(source, "    bool ensure(ID3D11Device* device,wchar_t const* path) {")
        self.assertNotIn("CreateTexture2D", setup)
        self.assertIn("if(table)return true;", setup)
        start = source.index("    bool prepare(ID3D11DeviceContext* context,")
        prepare = source[start:source.index("required_pages(receivers,basis)", start)]
        self.assertIn("ensure_field(device)", prepare)
        field = method(source, "    bool ensure_field(ID3D11Device* device){")
        run_cpp(r'''
#include <windows.h>
#include <d3d11.h>
#include <array>
#include <cassert>
#pragma comment(lib,"d3d11.lib")
struct Page {int x=0,y=0;};
struct Field {
 std::array<Page,32> pages{};ID3D11Texture2D* texture=nullptr;std::array<ID3D11RenderTargetView*,32> targets{};
 ID3D11ShaderResourceView* view=nullptr;
 template<class T> void drop(T*& p){if(p)p->Release();p=nullptr;}
''' + field + r'''
};
int main(){
 ID3D11Device* device=nullptr;
 if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,nullptr)))return 0;
 Field f;assert(f.ensure_field(device) && f.texture && f.view);
 D3D11_TEXTURE2D_DESC d{};f.texture->GetDesc(&d);
 assert(d.Width==1024 && d.Height==1024 && d.ArraySize==32 && d.Format==DXGI_FORMAT_R32_FLOAT);
 for(auto* target:f.targets)assert(target);
 auto* first=f.texture;assert(f.ensure_field(device) && f.texture==first); // created once
 f.drop(f.view);for(auto& t:f.targets)f.drop(t);f.drop(f.texture);device->Release();
 return 0;
}
''')


if __name__ == '__main__':
    unittest.main()
