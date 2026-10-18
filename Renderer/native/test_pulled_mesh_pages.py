"""Pulled mesh pages count resident bytes exactly when a run repeats an occurrence.

A run that drew one occurrence twice put it in one page group twice; the
second mapping released the new page before it was counted, so resident bytes
underflowed and trim retired every page on every frame (the user save:
2,391 retirements and 519 MB of copies for 101 entries).
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class PulledMeshPagesTests(unittest.TestCase):
    def test_repeated_occurrence_is_copied_once_and_counted(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <cassert>
#include <cstdio>
#include "Renderer/native/render_core/pulled_mesh_pages.h"
#pragma comment(lib,"d3d11.lib")
using c3x_renderer::render_core::PulledMeshPages;
int main(){
 Microsoft::WRL::ComPtr<ID3D11Device> device;Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)))return 2;
 float vertices[3*42]={};unsigned short indices[]={0,1,2};
 Microsoft::WRL::ComPtr<ID3D11Buffer> vb,ib;D3D11_BUFFER_DESC d{};D3D11_SUBRESOURCE_DATA data{};
 d.ByteWidth=sizeof(vertices);d.BindFlags=D3D11_BIND_VERTEX_BUFFER;data.pSysMem=vertices;
 if(FAILED(device->CreateBuffer(&d,&data,&vb)))return 3;
 d.ByteWidth=8;d.BindFlags=D3D11_BIND_INDEX_BUFFER;data.pSysMem=indices;
 if(FAILED(device->CreateBuffer(&d,&data,&ib)))return 4;
 PulledMeshPages::Source a{};a.buffer=vb.Get();a.indices=ib.Get();a.version=1;a.index_count=3;a.vertex_count=3;a.stride=168;a.index16=true;
 auto b=a;b.tx=128;
 PulledMeshPages pages;std::vector<PulledMeshPages::Range> ranges;
 PulledMeshPages::Source run[]={a,b,a};
 pages.begin_frame();assert(pages.resolve(device.Get(),context.Get(),run,3,ranges));
 // Two distinct occurrences, one page: 2 x 504 vertex bytes, 2 x 8 index bytes, 2 records.
 assert(pages.entry_count()==2 && pages.builds==1);
 assert(pages.bytes()==2*504+2*8+2*sizeof(PulledMeshPages::Record));
 unsigned drawn=0;for(auto const& r:ranges)drawn+=r.records;assert(drawn==3); // order preserved
 // A forced repack releases the old page exactly once; nothing underflows.
 pages.begin_frame();assert(pages.resolve(device.Get(),context.Get(),run,3,ranges,0));
 assert(pages.bytes()==2*504+2*8+2*sizeof(PulledMeshPages::Record) && pages.rebuilds==1);
 pages.begin_frame();assert(pages.resolve(device.Get(),context.Get(),run,3,ranges));
 assert(pages.builds==2 && pages.retired==0);
 std::puts("PASS pulled pages: repeated occurrence copied once, resident bytes exact across repack");
}
''', timeout=120)

    def test_pages_of_an_earlier_view_retire_after_idle_frames(self):
        """Pages stayed until the 512 MiB budget: the busy save kept about
        300 MB of jump-origin and scrolled-off pages (review 87). Pages no frame
        has drawn for idle_frames retire once residency passes idle_floor;
        pages drawn every frame stay, and a revisit rebuilds exactly."""
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <cassert>
#include <cstdio>
#include "Renderer/native/render_core/pulled_mesh_pages.h"
#pragma comment(lib,"d3d11.lib")
using c3x_renderer::render_core::PulledMeshPages;
int main(){
 Microsoft::WRL::ComPtr<ID3D11Device> device;Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)))return 2;
 float vertices[3*42]={};unsigned short indices[]={0,1,2};
 Microsoft::WRL::ComPtr<ID3D11Buffer> vb,ib;D3D11_BUFFER_DESC d{};D3D11_SUBRESOURCE_DATA data{};
 d.ByteWidth=sizeof(vertices);d.BindFlags=D3D11_BIND_VERTEX_BUFFER;data.pSysMem=vertices;
 if(FAILED(device->CreateBuffer(&d,&data,&vb)))return 3;
 d.ByteWidth=8;d.BindFlags=D3D11_BIND_INDEX_BUFFER;data.pSysMem=indices;
 if(FAILED(device->CreateBuffer(&d,&data,&ib)))return 4;
 PulledMeshPages::Source origin{};origin.buffer=vb.Get();origin.indices=ib.Get();origin.version=1;origin.index_count=3;origin.vertex_count=3;origin.stride=168;origin.index16=true;
 auto view=origin;view.tx=4096; // the view after a jump
 constexpr std::size_t page=504+8+sizeof(PulledMeshPages::Record);
 PulledMeshPages pages;pages.idle_floor=0;pages.idle_frames=100;std::vector<PulledMeshPages::Range> ranges;
 pages.begin_frame();assert(pages.resolve(device.Get(),context.Get(),&origin,1,ranges));
 assert(pages.entry_count()==1 && pages.bytes()==page);
 // The new view is drawn every frame; the origin stays only while recent.
 for(int frame=0;frame<99;++frame){pages.begin_frame();assert(pages.resolve(device.Get(),context.Get(),&view,1,ranges));}
 assert(pages.entry_count()==2 && pages.bytes()==2*page && pages.retired==0);
 for(int frame=0;frame<64;++frame){pages.begin_frame();assert(pages.resolve(device.Get(),context.Get(),&view,1,ranges));}
 assert(pages.entry_count()==1 && pages.bytes()==page && pages.retired==1 && pages.builds==2);
 // A residency at or below the floor keeps idle pages.
 PulledMeshPages kept;kept.idle_floor=4*page;kept.idle_frames=100;
 kept.begin_frame();assert(kept.resolve(device.Get(),context.Get(),&origin,1,ranges));
 for(int frame=0;frame<256;++frame){kept.begin_frame();assert(kept.resolve(device.Get(),context.Get(),&view,1,ranges));}
 assert(kept.entry_count()==2 && kept.retired==0);
 // Returning to the origin rebuilds its page with the same bounded copy.
 pages.begin_frame();assert(pages.resolve(device.Get(),context.Get(),&origin,1,ranges));
 assert(pages.entry_count()==2 && pages.builds==3 && ranges.size()==1 && ranges[0].records==1);
 // Many idle pages (a zoomed view's after zooming out) retire in one pass,
 // oldest first, down to the floor; the newest idle ones stay.
 PulledMeshPages many;many.idle_floor=4*page;many.idle_frames=100;
 for(int n=0;n<40;++n){auto s=origin;s.tx=n*256;s.ty=1;many.begin_frame();assert(many.resolve(device.Get(),context.Get(),&s,1,ranges));}
 assert(many.entry_count()==40 && many.bytes()==40*page);
 for(int frame=0;frame<200;++frame){many.begin_frame();assert(many.resolve(device.Get(),context.Get(),&view,1,ranges));}
 // The view page plus three of the newest idle pages fit the floor.
 assert(many.bytes()==4*page && many.entry_count()==4 && many.retired==37);
 std::puts("PASS pulled pages: idle pages of an earlier view retire, drawn pages stay");
}
''', timeout=120)


if __name__ == '__main__':
    unittest.main()
