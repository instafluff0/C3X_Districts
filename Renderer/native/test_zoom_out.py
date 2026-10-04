"""Zoom-out capture, HUD layout and native city-view configuration contracts."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_zoom_integration import function

ROOT = Path(__file__).resolve().parents[2]


class ZoomOutTests(unittest.TestCase):
    def test_capture_and_hud_keep_outer_tiles_reachable(self):
        source = (ROOT / 'injected_code.c').read_text()
        bounds = function(source, 'custom_renderer_capture_bounds')
        layout = function(source, 'custom_renderer_hud_layout_offset')
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include "Renderer/native/scene_projection.h"
struct RECT{int left,top,right,bottom;};
struct Main_Screen_Form {int camera_x=2600,camera_y=800,TileX_Min=0,TileX_Max=0,TileY_Min=0,TileY_Max=0;};
struct {struct{bool enable_custom_rendering=true,enable_custom_rendering_zoom=true;}current_config;
 int custom_renderer_zoom_target_width=128;}state,*is=&state;
struct {struct{struct{void* spotlight_on_city=nullptr;}Renderer;}Map;
 bool is_zoomed_out=false;int ScreenWidth=2240,ScreenHeight=1260;}bic,*p_bic_data=&bic;
bool custom_renderer_zoom_enabled(){return state.current_config.enable_custom_rendering&&
 state.current_config.enable_custom_rendering_zoom&&!bic.Map.Renderer.spotlight_on_city;}
RECT ''' + bounds + '\nvoid ' + layout + r'''
int main(){Main_Screen_Form screen;
 for(int width:{64,80,96,112})for(int enabled:{0,1,2,3}){
  state.custom_renderer_zoom_target_width=width;
  state.current_config.enable_custom_rendering=enabled!=0;
  state.current_config.enable_custom_rendering_zoom=enabled!=1;
  bic.Map.Renderer.spotlight_on_city=enabled==2?&screen:nullptr;
  screen.TileX_Min=91;auto captured=custom_renderer_capture_bounds(&screen);assert(screen.TileX_Min==91);
  if(enabled!=3){assert(screen.TileX_Min==91);int dx,dy;custom_renderer_hud_layout_offset(-800,1700,&dx,&dy);assert(!dx&&!dy);continue;}
  for(int x:{-1120,0,2240,3360})assert((screen.camera_x+x)/64>=captured.left&&(screen.camera_x+x)/64<=captured.right);
  for(int y:{-630,0,1260,1890})assert((screen.camera_y+y)/32>=captured.top&&(screen.camera_y+y)/32<=captured.bottom);
  for(int x:{-800,0,800,2240,3000})for(int y:{-300,600,1700}){
   int dx,dy;custom_renderer_hud_layout_offset(x,y,&dx,&dy);
   assert(x+dx>=256&&x+dx<=1984&&y+dy>=128&&y+dy<=1132);
   // Native ink at the safe layout anchor returns to the original world
   // position before the retained HUD applies the current display scale.
   for(float scale:{.5f,.625f,.75f,.875f,1.f,3.f}){
    c3x_renderer::SceneProjection p(2240,1260,scale);
    assert(std::abs((x+dx-dx+(x-1120)*(scale-1))-p.x(float(x)))<.001f);
   }
  }
 }
 state.custom_renderer_zoom_target_width=64;bic.is_zoomed_out=true;screen.TileX_Min=91;
 assert(custom_renderer_capture_bounds(&screen).left==91);bic.is_zoomed_out=false;
 state.custom_renderer_zoom_target_width=128;screen.TileX_Min=91;
 assert(custom_renderer_capture_bounds(&screen).left==91);assert(screen.TileX_Min==91);
}
''')

    def test_outer_hud_replays_from_safe_layout_at_fixed_size(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 constexpr unsigned w=64,h=48;Rect full={0,0,w,h};
 std::vector<unsigned> pixels(w*h),output;
 // Distinct background pixels require an actual present at each scale; a
 // flat background with off-screen ink legitimately returns an unchanged frame.
 for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)pixels[y*w+x]=0xff000000u|(x*3<<16)|(y*3<<8)|(x+y);
 D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
 desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 D3D11_SUBRESOURCE_DATA data={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> source;
 checked(device->CreateTexture2D(&desc,&data,&source));
 desc.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> display,buffer;
 checked(device->CreateTexture2D(&desc,nullptr,&display));checked(device->CreateTexture2D(&desc,nullptr,&buffer));
 ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
 for(auto anchor:std::vector<std::pair<int,int>>{{-8,24},{72,24},{32,-8},{32,56}}){
  Session session(device.Get(),context.Get());assert(session.publish(source.Get(),1));
  c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=1;
  c3x_renderer_gpu_result_v1 result={};
  auto create=[&](int format){request.action=C3X_GPU_CREATE;request.width=w;request.height=h;request.format=format;
   assert(session.execute(request,{}, {},result,output)==1);return Id(result.image);};
  auto map=create(C3X_GPU_RGB565),units=create(C3X_GPU_RGB565),ink=create(C3X_GPU_BGRA32);
  auto screen=create(C3X_GPU_RGB565),detail=create(C3X_GPU_BGRA32);
  auto send=[&](std::vector<Command> const& commands){request={};request.struct_size=sizeof(request);request.ticket=1;request.action=C3X_GPU_SUBMIT;
   assert(session.execute(request,commands,{},result,output)==1);};
  const int dx=32-anchor.first,dy=24-anchor.second;
  send({{Kind::quantize,map,session.map_image(),full,full},
        {Kind::fill,units,0,full,full,0,0,0x7c1f},{Kind::fill,ink,0,full,full,0,0,0xffff00ff},
        {Kind::hud_begin,units,0,{dx,dy,0,0},{},anchor.first,anchor.second,42,0,ink,0,0x7c1f},
        {Kind::fill,units,0,{30,23,35,26},full,0,0,0xffff},
        {Kind::fill,ink,0,{30,23,35,26},full,0,0,0xffffffff},{Kind::hud_end},
        {Kind::world_begin,screen,map,full,full,0,0,65536,0,detail,session.map_image(),int(w),int(h)},
        {Kind::world_end,screen,units,full,full,0,0,0x7c1f,0,detail,ink,int(w),int(h)}});
  assert(session.commit_display(1,detail,w,h,full));
  LARGE_INTEGER now={},frequency={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&now);int sample=0;long long tick=now.QuadPart;
  for(unsigned zoom:{32768u,40960u,49152u,57344u,65536u,32768u}){
   send({{Kind::zoom_target,0,0,{},{},0,0,zoom}});
   QueryPerformanceCounter(&now);tick=std::max(tick,now.QuadPart)+frequency.QuadPart;++sample;
   assert(session.visual_frame(tick,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())==1);
   session.did_present();if(session.presented_zoom()!=zoom)std::fprintf(stderr,"anchor %d,%d sample %d expected %u actual %u visual %.8f\n",anchor.first,anchor.second,sample,zoom,session.presented_zoom(),session.visual_scale());assert(session.presented_zoom()==zoom);
   auto actual=retained_read(device.Get(),context.Get(),display.Get());
   int x=anchor.first+int(std::lround((anchor.first-int(w/2))*(zoom/65536.f-1)));
   int y=anchor.second+int(std::lround((anchor.second-int(h/2))*(zoom/65536.f-1)));
   unsigned count=0;
   for(unsigned yy=0;yy<h;++yy)for(unsigned xx=0;xx<w;++xx){
    bool marked=int(xx)>=x-2&&int(xx)<x+3&&int(yy)>=y-1&&int(yy)<y+2;
    assert((actual[yy*w+xx]==0xffffffff)==marked);count+=marked;
   }
   if(zoom==32768)assert(count==15);
  }
 }
 std::puts("PASS outer HUD: unclipped 5x3 glyphs at all four edges across outward levels and reversal");
}
''', timeout=90)

    def test_city_auto_zoom_respects_effective_radius_and_configuration(self):
        source = (ROOT / 'injected_code.c').read_text()
        center = function(source, 'get_city_screen_center_y')
        focus = function(source, 'patch_Main_Screen_Form_bring_cnter_view_city_focus')
        run_cpp(r'''
#include <cassert>
#include <algorithm>
#include <initializer_list>
constexpr int __=0,WAL_CULTURAL_OR_ADJACENT=1;
struct City{struct{int Y=40;}Body;}city;
struct Main_Screen_Form{}screen;
struct{City* CurrentCity=&city;}form,*p_city_form=&form;
struct{bool is_zoomed_out=false;int ScreenWidth=2240,ScreenHeight=1260;}bic,*p_bic_data=&bic;
struct c3x_renderer_tile_v1{int tile_x=60,tile_y=40,anchor_x=0,anchor_y=0;}captured;
struct{struct{int city_work_radius=2,work_area_limit=0;
 bool auto_zoom_city_screen_for_large_work_areas=true,enable_custom_rendering=true;}current_config;}state,*is=&state;
int limit=2,calls=0;
int not_above(int a,int b){return std::min(a,b);}
int get_work_ring_limit_total(City* c){assert(c==&city);return limit;}
void Main_Screen_Form_bring_tile_into_view(Main_Screen_Form* p,int,int x,int y,int reason,bool update,bool force){
 assert(p==&screen&&x==60&&reason==7&&update&&!force);++calls;
 assert(y>=city.Body.Y);}
int ''' + center + '\nvoid ' + focus + r'''
int main(){for(bool renderer:{false,true})for(bool auto_zoom:{false,true})for(int radius:{2,3,4,5})
 for(limit=1;limit<=5;++limit)for(int policy:{0,1}){
  auto& c=state.current_config;c.enable_custom_rendering=renderer;c.auto_zoom_city_screen_for_large_work_areas=auto_zoom;
  c.city_work_radius=radius;c.work_area_limit=policy;bic.is_zoomed_out=false;
  int effective=std::min(radius,limit);if(policy)effective=std::min(radius,effective+1);
  auto before=calls;patch_Main_Screen_Form_bring_cnter_view_city_focus(&screen,19,60,42,7,true,false);
  assert(calls==before+1&&bic.is_zoomed_out==(auto_zoom&&effective>=4));
 }
}
''')

    def test_city_z_uses_only_native_half_and_normal_view(self):
        source = (ROOT / 'injected_code.c').read_text()
        body = function(source, 'patch_City_Form_m82_handle_key_event')
        # Keep the production Z branch and native forwarding verbatim; the
        # preceding AI production-ranking branch is unrelated to zoom.
        body = body[body.index('} else if (is->current_config.toggle_zoom_with_z_on_city_screen'):]
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
constexpr int __=0,VK_Z=90;
struct City{struct{int X=60,Y=40,ID=7;}Body;}city;
struct Base_Form{};
int redraws=0,moves=0,native=0;
int int_abs(int x){return x<0?-x:x;}
struct Table{void(*m73_call_m22_Draw)(Base_Form*);};
void draw(Base_Form*){++redraws;}Table table{draw};
struct City_Form{City* CurrentCity=&city;struct{Table* vtable=&table;}Base;};
struct Main_Screen_Form{int camera_x=0,camera_y=0;}screen,*p_main_screen_form=&screen;
struct{bool is_zoomed_out=false;int ScreenWidth=2240,ScreenHeight=1260;}bic,*p_bic_data=&bic;
struct c3x_renderer_tile_v1{int tile_x=60,tile_y=40,anchor_x=0,anchor_y=0;}captured;
struct{struct{bool enable_custom_rendering=true,toggle_zoom_with_z_on_city_screen=true;}current_config;
 bool custom_renderer_trace_input=false;int custom_renderer_zoom_target_width=64;
 int custom_renderer_tile_count=1;c3x_renderer_tile_v1* custom_renderer_tiles=&captured;}state,*is=&state;
int get_city_screen_center_y(City* c){return c->Body.Y+2;}
void Main_Screen_Form_bring_tile_into_view(Main_Screen_Form*,int,int,int,int,bool,bool){++moves;
 int width=bic.is_zoomed_out?64:128;captured.anchor_x=1120-width/2;captured.anchor_y=566-width/4;}
void Main_Screen_Form_tile_to_screen_coords(Main_Screen_Form*,int,int,int,int*x,int*y){*x=*y=0;}
int traces=0;void debug(char const* line){assert(std::strstr(line,"city_anchor=1120,566"));++traces;}auto p_OutputDebugStringA=debug;
void City_Form_m82_handle_key_event(City_Form*,int,int,int){++native;}
void invoke(City_Form* self,int virtual_key_code,int is_down){if(false){
''' + body + r'''
int main(){City_Form form;for(bool renderer:{false,true})for(int world_zoom:{64,80,112,128,384}){
 state.current_config.enable_custom_rendering=renderer;state.custom_renderer_zoom_target_width=world_zoom;
 for(bool enabled:{false,true}){state.current_config.toggle_zoom_with_z_on_city_screen=enabled;bic.is_zoomed_out=false;
  for(int n=0;n<10;++n){auto before=moves;invoke(&form,VK_Z,1);
   assert(bic.is_zoomed_out==(enabled&&(n%2==0)));assert(moves==before+int(enabled));
   bool zoom=bic.is_zoomed_out;invoke(&form,VK_Z,0);invoke(&form,88,1);assert(bic.is_zoomed_out==zoom);
   assert(state.custom_renderer_zoom_target_width==world_zoom);
  }
 }}assert(native==600&&redraws==moves);
 state.custom_renderer_trace_input=true;state.current_config.enable_custom_rendering=true;
 state.current_config.toggle_zoom_with_z_on_city_screen=true;
 invoke(&form,VK_Z,1);invoke(&form,VK_Z,1);assert(traces==2);}
''')


if __name__ == '__main__':
    unittest.main()
