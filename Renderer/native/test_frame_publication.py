"""Execute the production publication owner and its failure/size boundaries."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PublicationTests(unittest.TestCase):
    def test_terrain_preview_anchors_ownership_and_input_bounds(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        body = "struct PublishedMapFrame {" + source.split("struct PublishedMapFrame {", 1)[1].split("// Civ III remains", 1)[0]
        program = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/environment_runtime.h"
''' + body + r'''
struct Texture {bool configured=true;std::vector<std::uint8_t> dds;};
void word(std::vector<std::uint8_t>& data,unsigned at,unsigned value){for(unsigned c=0;c<4;++c)data[at+c]=std::uint8_t(value>>(8*c));}
std::vector<std::uint8_t> texture(unsigned color){
    std::vector<std::uint8_t> dds(164);std::memcpy(dds.data(),"DDS ",4);word(dds,4,124);
    word(dds,12,4);word(dds,16,4);std::memcpy(dds.data()+84,"DX10",4);word(dds,128,78);
    word(dds,156,color|(color<<16));return dds;
}
int main(){
    std::array<Texture,14> textures;
    textures[2].dds=texture(0xf800);textures[1].dds=texture(0x07e0);
    c3x_renderer_tile_v1 tiles[3]={};
    for(auto& t:tiles){t.tile_flags=C3X_RENDERER_TILE_RENDER;t.terrain_type=2;t.real_terrain_type=2;}
    tiles[1].anchor_x=64;tiles[1].terrain_type=tiles[1].real_terrain_type=1;
    tiles[2].anchor_x=32;tiles[2].anchor_y=16;tiles[2].tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
    c3x_renderer_frame_v1 frame={};frame.target_width=128;frame.target_height=64;frame.tile_width=64;frame.tile_height=32;
    frame.tile_count=3;frame.tiles=tiles;frame.hour=12;frame.clip_right=128;frame.clip_bottom=64;
    CameraTerrainPreview preview;PublishedMapFrame out;std::atomic<bool> cancelled{false};
    assert(preview.render(frame,textures,out,cancelled));
    auto red=out.pixels[16*128+32],green=out.pixels[16*128+96];
    assert((red&0x00ff0000)!=0 && (red&0x0000ffff)==0);
    assert((green&0x0000ff00)!=0 && (green&0x00ff00ff)==0);
    assert(out.pixels[32*128+64]==0xff000000u); // Halo cannot disclose terrain.
    assert(out.output.rendered_tile_count==2 && out.output.replacement_tile_count==3);
    assert(out.replacements[0]==C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED && out.replacements[1]==out.replacements[0] && out.replacements[2]==0);
    assert(out.output.bgra_pixels==out.pixels.data() && out.output.replacement_tile_flags==out.replacements.data());
    frame.hour=0;assert(preview.render(frame,textures,out,cancelled));assert(out.pixels[16*128+32]!=red);
    frame.hour=12;
    for(auto& t:tiles)t.anchor_x+=32;
    assert(preview.render(frame,textures,out,cancelled));
    assert(out.pixels[16*128+64]==red && out.pixels[16*128+32]==0xff000000u);
    frame.tile_width=128;frame.tile_height=64;tiles[0].anchor_x=0;tiles[1].tile_flags=0;
    assert(preview.render(frame,textures,out,cancelled));assert(out.pixels[32*128+64]==red);
    auto preserved=out.pixels;
    textures[2].dds.resize(163);assert(!preview.render(frame,textures,out,cancelled));assert(out.pixels==preserved);
    textures[2].dds=texture(0xf800);word(textures[2].dds,128,71);assert(!preview.render(frame,textures,out,cancelled));
    textures[2].dds=texture(0xf800);word(textures[2].dds,16,16385);assert(!preview.render(frame,textures,out,cancelled));
    textures[2].dds=texture(0xf800);textures[2].configured=false;assert(!preview.render(frame,textures,out,cancelled));
    textures[2].configured=true;cancelled=true;assert(!preview.render(frame,textures,out,cancelled));cancelled=false;
    frame.target_width=8192;frame.target_height=8192;assert(!preview.render(frame,textures,out,cancelled));assert(out.pixels==preserved);
    frame.target_width=128;frame.target_height=64;tiles[0].anchor_x=INT_MAX;tiles[0].anchor_y=INT_MIN;
    assert(preview.render(frame,textures,out,cancelled));for(auto p:out.pixels)assert(p==0xff000000u);
}
'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preview.cpp"
            path.write_text(program)
            exe = Path(folder) / "preview"
            built = subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(path), str(ROOT / "Renderer/native/environment_runtime.cpp"), "-o", str(exe)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            subprocess.run([str(exe)], check=True, timeout=30)

    def test_camera_cancellation_reaches_draw_submission(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        submission = source.split('trace.write("geometry-ready", frame_cache_path);', 1)[1].split('trace.write("readback-begin"', 1)[0]
        self.assertIn("c3x_renderer::power_of_two_extent(height),foreground_pending)", submission)
        self.assertEqual(submission.count("if(cancelled())return false;"), 2)

    def test_actual_worker_camera_supersession_and_takeover(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        publication = "struct PublishedMapFrame {" + source.split("struct PublishedMapFrame {", 1)[1].split("// Civ III remains", 1)[0]
        worker = "class RendererWorker {" + source.split("class RendererWorker {", 1)[1].split("RendererWorker * renderer_worker", 1)[0]
        program = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/environment_runtime.h"
using HDC=void*;
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* out){out->QuadPart=std::chrono::steady_clock::now().time_since_epoch().count();}
unsigned GetEnvironmentVariableA(char const*,char*,std::size_t){return 0;}
template<std::size_t N,class... T> void sprintf_s(char (&buffer)[N],char const* format,T... args){std::snprintf(buffer,N,format,args...);}
namespace c3x_renderer {
struct Signature {std::uint64_t complete=0;};
Signature terrain_frame_signature(c3x_renderer_frame_v1 const& f,long long,unsigned){
    return {std::uint64_t(f.presentation_time_ticks)+1};
}
}
struct Trace {int level=0;void write(char const*,char const*,bool=false){} double milliseconds(long long value){return double(value)/1000000;}};
struct Footprint {int coordinate=0;struct {int left=0,right=0;} bounds;};
struct Bodies {
    char const* failure_reason="";bool cache_hit=false;std::size_t cache_bytes=0;unsigned keyed_pixels=0,cast_pixels=0;
    template<class F> bool render(int,int,c3x_renderer_unit_v1 const&,F){return true;}
    bool blit(HDC,int,int,HDC){return true;}void reset_gpu(){}
};
struct RendererState {
    struct Terrain {bool configured=false;std::vector<std::uint8_t> dds;};
    std::array<Terrain,14> terrain_textures;
    Trace trace;Bodies unit_bodies;bool unit_rendering_enabled=true,pickup_profile=false,cache_valid=false;
    int device=0,context=0;
    unsigned cache_hits=0,device_recoveries=0,frame_tiles_built=0,prepared_blocks=0,visible_resource_animations=0;
    std::size_t prefetched_geometry_bytes=0,tile_geometry_cache_bytes=0;
    std::uint64_t requested_signature=0;
    struct {std::size_t bytes=0;} pixel_blocks;
    struct {void clear(){}} geometry_cache;
    std::vector<Footprint> pixel_neighborhood;Footprint prepared_footprint;
    std::vector<c3x_renderer_tile_v1> cached_tiles;
    std::vector<unsigned> pixels,flags;
    std::atomic<unsigned> entered{0},cancelled{0},resets{0};
    std::atomic<bool> hold{false};
    bool render(c3x_renderer_frame_v1 const& f,c3x_renderer_output_v1& out,int=-1,
                std::atomic<bool> const* stop=nullptr,std::uint64_t=0){
        ++entered;
        while(hold.load()){
            if(stop && stop->load()){++cancelled;return false;}
            std::this_thread::yield();
        }
        if(stop && stop->load()){++cancelled;return false;}
        unsigned value=unsigned(f.tiles[0].anchor_x)^f.world_topology[0];
        pixels.assign(std::size_t(f.target_width)*f.target_height,value);
        flags.assign(f.tile_count,f.tiles[0].tile_flags);
        cached_tiles.assign(f.tiles,f.tiles+f.tile_count);
        requested_signature=std::uint64_t(f.presentation_time_ticks)+1;
        out={C3X_RENDERER_API_VERSION,sizeof(out)};
        out.width=f.target_width;out.height=f.target_height;out.stride_bytes=out.width*4;
        out.bgra_pixels=pixels.data();out.replacement_tile_flags=flags.data();out.replacement_tile_count=f.tile_count;
        out.clip_right=out.width;out.clip_bottom=out.height;return true;
    }
    static long long resource_clock(c3x_renderer_frame_v1 const&){return 0;}
    bool configure_pack(char const*){reset();return true;}
    bool configure_definitions(char const*,char const*,char const*,char const*){reset();return true;}
    void reset(){++resets;pixels.clear();flags.clear();}
    void clear_geometry_vertex_buffers(){}
    template<class T> bool prepare_unit_action(T const&){return true;}
    void begin_pixel_neighborhood(c3x_renderer_frame_v1 const&){}
    void start_pixel_preparation(){}void cancel_pixel_preparation(){}
    unsigned pixel_work_pending(){return 0;}
    bool prepare_pixel_block(std::atomic<bool> const&){return false;}
};
struct MapBlitter {
    std::thread::id ui=std::this_thread::get_id();
    template<class T> bool blit(c3x_renderer_output_v1 const&,HDC,int,int,T&){assert(std::this_thread::get_id()==ui);return true;}
    void reset_blit_surface(){assert(std::this_thread::get_id()==ui);}
};
''' + publication + worker + r'''
template<class Predicate> void until(Predicate ready){
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(!ready()){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
}
int main(){
    RendererState state;RendererWorker worker(state);
    c3x_renderer_tile_v1 tile={};tile.tile_flags=C3X_RENDERER_TILE_RENDER;tile.anchor_x=7;
    unsigned topology=42;
    c3x_renderer_frame_v1 f={};f.target_width=2240;f.target_height=1192;f.tile_width=128;f.tile_height=64;
    f.tile_count=1;f.tiles=&tile;f.world_topology=&topology;f.world_topology_count=1;
    f.presentation_time_ticks=1;
    c3x_renderer_output_v1 out={C3X_RENDERER_API_VERSION,sizeof(out)};
    assert(worker.render(f,out)==C3X_RENDERER_RESULT_OK);
    c3x_renderer_i64 first=0,last=0;
    assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return worker.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
    auto old=out;auto old_value=7u^42u;
    state.hold=true;auto entered=state.entered.load();
    assert(worker.camera_begin(f,first)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    // Latest-wins under actual worker contention, not a detached queue model.
    for(unsigned i=0;i<30;++i){tile.anchor_x=int(100+i);topology=1000+i;++f.presentation_time_ticks;
        assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);}
    auto expected=unsigned(tile.anchor_x)^topology;
    tile.anchor_x=-999;topology=0; // Caller may immediately reuse capture storage.
    assert(worker.camera_poll(first,out)==C3X_RENDERER_RESULT_SUPERSEDED);
    assert(std::memcmp(&old,&out,sizeof(out))==0);
    assert(worker.blit(old,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
    auto old_pixels=static_cast<unsigned const*>(old.bgra_pixels);
    for(std::size_t i=0;i<2240u*1192u;++i)assert(old_pixels[i]==old_value);
    state.hold=false;
    until([&]{return worker.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
    auto pixels=static_cast<unsigned const*>(out.bgra_pixels);
    for(std::size_t i=0;i<2240u*1192u;++i)assert(pixels[i]==expected);
    assert(out.replacement_tile_count==1 && out.replacement_tile_flags[0]==C3X_RENDERER_TILE_RENDER);
    assert(state.cancelled>0 && state.resets==0); // Supersession is not device recovery.
    // Synchronous unit work cancels an actively blocked terrain build first.
    state.hold=true;entered=state.entered.load();
    assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    c3x_renderer_unit_v1 unit={};
    assert(worker.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
    assert(worker.camera_poll(last,out)==C3X_RENDERER_RESULT_SUPERSEDED);
    // Reset also joins an active cancellation, and never hangs on a pending job.
    entered=state.entered.load();assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    worker.reset_and_stop();assert(state.resets==1);
    assert(worker.camera_poll(last,out)==C3X_RENDERER_RESULT_SUPERSEDED);
}
'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "worker.cpp"
            path.write_text(program)
            exe = Path(folder) / "worker"
            built = subprocess.run([compiler, "-std=c++17", "-O1", "-pthread", "-I", str(ROOT), str(path), str(ROOT / "Renderer/native/environment_runtime.cpp"), "-o", str(exe)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            subprocess.run([str(exe)], check=True, timeout=30)

    def test_ui_blitter_resource_lifetime_and_failures(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        render_state = source.split("class RendererState {", 1)[1].split("class MapBlitter {", 1)[0]
        self.assertNotIn("DeleteDC(", render_state)
        self.assertNotIn("reset_blit_surface", render_state)
        body = "class MapBlitter {" + source.split("class MapBlitter {", 1)[1].split("RendererState renderer;", 1)[0]
        program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/color_quantization.h"
using HGDIOBJ=void*;
struct Bitmap {int width=1,height=1;std::vector<std::uint32_t> pixels;};
using HBITMAP=Bitmap*;
Bitmap stock;
struct DC {HBITMAP selected=&stock;};
using HDC=DC*;
struct BITMAPINFOHEADER {int biSize=0,biWidth=0,biHeight=0,biPlanes=0,biBitCount=0,biCompression=0;};
struct BITMAPINFO {BITMAPINFOHEADER bmiHeader;};
struct BITMAP {int bmBitsPixel=32;};
struct DIBSECTION {BITMAP dsBm;BITMAPINFOHEADER dsBmih;unsigned dsBitfields[3]={};};
constexpr int BI_RGB=0,DIB_RGB_COLORS=0,OBJ_BITMAP=7,SRCCOPY=1,FALSE=0;
HGDIOBJ const HGDI_ERROR=reinterpret_cast<HGDIOBJ>(std::uintptr_t(-1));
int failure=0,bitmaps=0;
std::vector<HDC> contexts;
auto const ui_thread=std::this_thread::get_id();
void on_ui(){assert(std::this_thread::get_id()==ui_thread);}
bool fail(int stage){if(failure!=stage)return false;failure=0;return true;}
HDC CreateCompatibleDC(void*){on_ui();if(fail(1))return nullptr;auto dc=new DC;contexts.push_back(dc);return dc;}
HBITMAP CreateDIBSection(HDC,BITMAPINFO const* info,int,void** bits,void*,int){
    on_ui();if(fail(2))return nullptr;auto bitmap=new Bitmap;
    bitmap->width=info->bmiHeader.biWidth;bitmap->height=-info->bmiHeader.biHeight;
    bitmap->pixels.resize(bitmap->width*bitmap->height);*bits=bitmap->pixels.data();++bitmaps;return bitmap;
}
HGDIOBJ SelectObject(HDC dc,HGDIOBJ object){on_ui();if(fail(3))return nullptr;if(fail(5))return HGDI_ERROR;auto old=dc->selected;dc->selected=static_cast<HBITMAP>(object);return old;}
void DeleteObject(HBITMAP bitmap){on_ui();assert(bitmap!=&stock);for(auto dc:contexts)assert(dc->selected!=bitmap);delete bitmap;--bitmaps;}
void DeleteDC(HDC dc){on_ui();assert(dc->selected==&stock);contexts.erase(std::find(contexts.begin(),contexts.end(),dc));delete dc;}
HGDIOBJ GetCurrentObject(HDC dc,int){on_ui();return dc->selected;}
int GetObjectA(HGDIOBJ,int,DIBSECTION* section){on_ui();*section={};return sizeof(DIBSECTION);}
int BitBlt(HDC destination,int x,int y,int width,int height,HDC source,int sx,int sy,int){
    on_ui();if(fail(4))return FALSE;
    for(int row=0;row<height;++row)std::memcpy(destination->selected->pixels.data()+(y+row)*destination->selected->width+x,
        source->selected->pixels.data()+(sy+row)*source->selected->width+sx,std::size_t(width)*4);
    return 1;
}
struct Trace {void write(char const*,char const*,bool){}};
''' + body + r'''
int main(){
    HDC destination=CreateCompatibleDC(nullptr);BITMAPINFO info={};
    info.bmiHeader.biWidth=64;info.bmiHeader.biHeight=-64;void* bits=nullptr;
    auto target=CreateDIBSection(destination,&info,0,&bits,nullptr,0);SelectObject(destination,target);
    std::vector<std::uint32_t> pixels(32*32,0x12345678);Trace trace;
    c3x_renderer_output_v1 out={};out.width=out.height=16;out.stride_bytes=64;
    out.bgra_pixels=pixels.data();out.clip_left=out.clip_top=2;out.clip_right=out.clip_bottom=14;
    {
        MapBlitter blitter;
        for(int width:{8,16,32,16,8}){
            out.width=out.height=width;out.stride_bytes=width*4;out.clip_right=out.clip_bottom=width-2;
            std::fill(target->pixels.begin(),target->pixels.end(),0xabcdef01);
            assert(blitter.blit(out,destination,-17,5,trace));
            assert(bitmaps==2 && contexts.size()==2);
            for(int y=0;y<64;++y)for(int x=0;x<64;++x){
                bool inside=x>=2 && x<width-2 && y>=2 && y<width-2;
                assert(target->pixels[y*64+x]==(inside?0x12345678:0xabcdef01));
            }
        }
        blitter.reset_blit_surface();blitter.reset_blit_surface();
        assert(bitmaps==1 && contexts.size()==1);
        assert(blitter.blit(out,destination,0,0,trace));
    }
    assert(bitmaps==1 && contexts.size()==1);
    for(int stage:{1,2,3,4,5}){
        {
            MapBlitter blitter;failure=stage;
            assert(!blitter.blit(out,destination,0,0,trace));assert(failure==0);
            assert(blitter.blit(out,destination,0,0,trace));
        }
        assert(bitmaps==1 && contexts.size()==1);
    }
    SelectObject(destination,&stock);DeleteObject(target);DeleteDC(destination);
    assert(!bitmaps && contexts.empty());
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-blitter-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_owned_output_atomic_failure_and_budget(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        body = "struct PublishedMapFrame {" + source.split("struct PublishedMapFrame {", 1)[1].split(
            "// Cheap, deliberately provisional", 1)[0]
        program = r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
int allocations_before_failure=-1;
void* operator new(std::size_t size) {
    if(allocations_before_failure==0)throw std::bad_alloc();
    if(allocations_before_failure>0)--allocations_before_failure;
    if(auto p=std::malloc(size?size:1))return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
''' + body + r'''
int main() {
    std::vector<std::uint32_t> image(2240*1192);
    for(unsigned i=0;i<image.size();++i)image[i]=i*193u;
    std::uint32_t fallback[]={3,7},flags[]={1,2,3,4,5,6,7,8};
    c3x_renderer_output_v1 out={};
    out.api_version=C3X_RENDERER_API_VERSION;out.struct_size=sizeof(out);
    out.width=2240;out.height=1192;out.stride_bytes=2240*4;
    out.clip_right=2240;out.clip_bottom=1192;out.bgra_pixels=image.data();
    out.fallback_tile_count=2;out.fallback_tile_indices=fallback;
    out.replacement_tile_count=8;out.replacement_tile_flags=flags;
    out.content_revision=17;out.device_generation=23;out.visible_animation_count=2;
    PublishedMapFrame published;
    assert(published.capture(out,-73,119));
    assert(published.phase_x==-73 && published.phase_y==119);
    assert(published.output.content_revision==17 && published.output.device_generation==23);
    assert(published.output.visible_animation_count==2);
    assert(published.output.bgra_pixels!=out.bgra_pixels);
    assert(published.output.fallback_tile_indices!=fallback);
    assert(published.output.replacement_tile_flags!=flags);
    assert(!std::memcmp(published.output.bgra_pixels,image.data(),image.size()*4));
    assert(published.output.fallback_tile_indices[1]==7 && published.output.replacement_tile_flags[7]==8);
    image[100]=0;fallback[1]=0;flags[7]=0;
    assert(published.pixels[100]==19300 && published.fallback[1]==7 && published.replacements[7]==8);
    auto old_pixels=published.output.bgra_pixels;
    for(int failure=0;failure<3;++failure){
        allocations_before_failure=failure;
        assert(!published.capture(out,0,0));allocations_before_failure=-1;
        assert(published.output.bgra_pixels==old_pixels && published.pixels[100]==19300);
        assert(published.phase_x==-73 && published.replacements[7]==8);
    }
    for(int width:{0,-1,8193,8192}){
        auto bad=out;bad.width=width;bad.height=8192;bad.stride_bytes=width*4;
        bad.bgra_pixels=reinterpret_cast<void const*>(1);
        assert(!published.capture(bad,0,0));assert(published.output.bgra_pixels==old_pixels);
    }
    auto bad=out;bad.replacement_tile_flags=nullptr;assert(!published.capture(bad,0,0));
    bad=out;bad.fallback_tile_indices=nullptr;assert(!published.capture(bad,0,0));
    bad=out;bad.replacement_tile_count=8193;assert(!published.capture(bad,0,0));
    bad=out;bad.stride_bytes=1;assert(!published.capture(bad,0,0));
    assert(published.capture(published.output,7,-9)); // Aliased input is copied before commit.
    assert(published.pixels[100]==19300 && published.phase_x==7 && published.phase_y==-9);
    out.fallback_tile_count=out.replacement_tile_count=0;
    out.fallback_tile_indices=out.replacement_tile_flags=nullptr;
    assert(published.capture(out,1,2));
    assert(published.output.fallback_tile_indices==nullptr && published.output.replacement_tile_flags==nullptr);
    assert(published.pixels[100]==0 && published.phase_x==1 && published.phase_y==2);
    published.clear();assert(!published.output.bgra_pixels && published.pixels.capacity()==0);
    assert(published.fallback.capacity()==0 && published.replacements.capacity()==0);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-publication-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
