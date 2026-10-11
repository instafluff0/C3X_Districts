"""A settled zoomed step is drawn once, at the presented zoom (review 69).

The hidden canonical (1x) image used to be a second full scene render on every
zoomed step: 36-80 ms of its water pass alone on the busy save, because the
canonical lane's caches are cold while zoomed. Everything that reads it while
zoomed samples the centred rectangle the zoomed draw covers, so the step is
drawn at the presented zoom and the canonical centre is that draw minified.
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

NATIVE = ROOT / "Renderer/native"


class ZoomedStepCanonicalTests(unittest.TestCase):
    def test_minified_zoomed_draw_reproduces_the_canonical_centre(self):
        # A zoomed draw of piecewise-constant content replicates each canonical
        # pixel over its SceneProjection footprint; minification must return
        # exactly the original pixels, and write nothing outside the region.
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "Renderer/native/render_core/canonical_minify.h"
#include "Renderer/native/scene_projection.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)))return 2;
 unsigned const w=64,h=48;std::uint32_t const sentinel=0xff00ff00u;
 std::vector<std::uint32_t> canonical(w*h);
 for(unsigned i=0;i<w*h;++i)canonical[i]=0xff000000u|((i*2654435761u)>>8&0xffffffu);
 for(float zoom:{2.f,3.f}){
  c3x_renderer::SceneProjection projection(w,h,zoom);
  // Zoomed texel d shows canonical pixel floor(c + (d + .5 - c) / zoom).
  std::vector<std::uint32_t> zoomed(w*h);
  for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
   int qx=int(std::floor(projection.cx+(x+.5f-projection.cx)/zoom)),qy=int(std::floor(projection.cy+(y+.5f-projection.cy)/zoom));
   zoomed[y*w+x]=canonical[unsigned(qy)*w+unsigned(qx)];}
  D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
  d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA data={zoomed.data(),w*4,0};
  ComPtr<ID3D11Texture2D> source,target,readback;ComPtr<ID3D11ShaderResourceView> view;ComPtr<ID3D11RenderTargetView> rtv;
  if(FAILED(device->CreateTexture2D(&d,&data,&source))||FAILED(device->CreateShaderResourceView(source.Get(),nullptr,&view)))return 3;
  std::vector<std::uint32_t> fill(w*h,sentinel);data.pSysMem=fill.data();
  if(FAILED(device->CreateTexture2D(&d,&data,&target))||FAILED(device->CreateRenderTargetView(target.Get(),nullptr,&rtv)))return 4;
  d.BindFlags=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  if(FAILED(device->CreateTexture2D(&d,nullptr,&readback)))return 5;
  c3x_renderer::render_core::CanonicalMinify minify;
  if(!minify.draw(device.Get(),context.Get(),view.Get(),rtv.Get(),w,h,zoom))return 6;
  context->CopyResource(readback.Get(),target.Get());
  D3D11_MAPPED_SUBRESOURCE mapped={};if(FAILED(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped)))return 7;
  auto region=c3x_renderer::render_core::CanonicalMinify::region(w,h,zoom);
  unsigned exact=0;
  for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
   auto got=static_cast<std::uint32_t const*>(mapped.pData)[y*mapped.RowPitch/4+x];
   bool inside=int(x)>=region.left&&int(x)<region.right&&int(y)>=region.top&&int(y)<region.bottom;
   if(!inside){if(got!=sentinel){std::printf("FAIL zoom=%.0f outside %u,%u written\n",zoom,x,y);return 8;}continue;}
   // Footprint [c + (q - c) zoom, c + (q + 1 - c) zoom) wholly inside the zoomed image.
   float l=projection.x(float(x)),t=projection.y(float(y)),r=projection.x(float(x+1)),b=projection.y(float(y+1));
   if(l<0||t<0||r>float(w)||b>float(h)){if(got==sentinel){std::printf("FAIL zoom=%.0f edge %u,%u unwritten\n",zoom,x,y);return 9;}continue;}
   if(got!=canonical[y*w+x]){std::printf("FAIL zoom=%.0f at %u,%u got %08x want %08x\n",zoom,x,y,got,canonical[y*w+x]);return 10;}
   ++exact;}
  context->Unmap(readback.Get(),0);
  unsigned span_x=unsigned(std::ceil(w/zoom)),span_y=unsigned(std::ceil(h/zoom));
  if(exact<(span_x-2)*(span_y-2)){std::printf("FAIL zoom=%.0f only %u exact\n",zoom,exact);return 11;}
  std::printf("zoom=%.0f region=%d,%d,%d,%d exact=%u\n",zoom,region.left,region.top,region.right,region.bottom,exact);
 }
 std::puts("PASS canonical minify: the zoomed draw's minification is the canonical centre");
}
''', timeout=120)

    def test_settled_zoomed_steps_draw_once_at_the_presented_zoom(self):
        source = (NATIVE / "c3x_renderer.cpp").read_text()
        deferred = source[source.index("bool draw_deferred_step(){"):]
        deferred = deferred[:deferred.index("void prepare_ahead(")]
        # The zoom is settled (destination equals the presented zoom), zoomed
        # in, and the control restores the canonical draw.
        settled = source[source.index("float settled_zoom(){"):source.index("bool draw_deferred_step(){")]
        self.assertRegex(settled, r"presented>1\.f && std::fabs\(destination-presented\)<1e-4f")
        self.assertIn("C3X_RENDERER_ZOOMED_STEP_CONTROL", settled)
        self.assertIn("renderer_state.zoomed_step=deferred_step_zoom=settled_zoom();", deferred)
        # A camera job that moved the camera (its retained view retired, so no
        # display frame runs during it) is drawn at the settled zoom too.
        job = source[source.index("else if(!initial_world && defer_scroll_step(ready))result=C3X_RENDERER_RESULT_OK;"):]
        job = job[:job.index("bool complete=renderer_state.render(job_frame")]
        self.assertIn("if(!initial_world && !completed_scene_usable())renderer_state.zoomed_step=settled_zoom();", job)
        self.assertIn("struct ZoomLease {float& value;~ZoomLease(){value=1.f;}} zoom_lease{renderer_state.zoomed_step};", job)
        # A full camera job's canonical draw is not redrawn at 1x on adoption
        # while the zoom is settled: the zoomed draw follows it.
        self.assertIn("if(map_sample.prepare && !drawn_now && settled_zoom()==1.f){", source)
        self.assertLess(deferred.index("renderer_state.zoomed_step="), deferred.index("auto draw=[this]"))
        step = source[source.index("route_map_serial=gpu_serial+1;\n            bool drawn=false;"):]
        step = step[:step.index("fill_output(frame,output,invalidations")]
        # One scene render: at the presented zoom, then the minification into
        # the canonical target; the 1x render only when not zoomed.
        self.assertIn("c3x_renderer64_render_fresh(native_frame,zoomed_step_target,zoomed_step) &&\n"
                      "                    canonical_minify.draw(device,context,zoomed_step_view,target,", step)
        self.assertIn("}else drawn=c3x_renderer64_render_fresh(native_frame,target);", step)
        self.assertEqual(len(re.findall(r"c3x_renderer64_render_fresh\(", step)), 2)
        # Units are selected for the zoom actually drawn.
        self.assertIn("select_frame_units(native_frame,candidates,fresh_unit_poses,zoomed_step)", source)
        # That draw becomes the new map's first zoomed front instead of being
        # drawn again after publication; the next step gets a new target.
        adoption = source[source.index("bool zoomed_front=drawn_now && ready && deferred_step_zoom!=1.f"):]
        adoption = adoption[:adoption.index("}else result=C3X_RENDERER_RESULT_BAD_ARGUMENT;")]
        self.assertIn("prepared_map->front=renderer_state.zoomed_step_texture;prepared_map->zoom=deferred_step_zoom;", adoption)
        self.assertIn("renderer_state.release(renderer_state.zoomed_step_texture);", adoption)
        self.assertIn("if(zoomed_prepare&&moved&&!zoomed_front&&presented!=1.f", adoption)


if __name__ == '__main__':
    unittest.main()
