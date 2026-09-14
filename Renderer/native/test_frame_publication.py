"""Execute the production publication owner and its failure/size boundaries."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class PublicationTests(unittest.TestCase):
    def test_memory_tier_shrink_retires_optional_owners(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        body = "bool three_zoom_memory=" + source.split("bool three_zoom_memory=", 1)[1].split("clip_dirty_blocks=", 1)[0]
        program = r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>
#include "Renderer/native/navigation_options.h"
constexpr std::size_t mib=1024*1024;
constexpr std::size_t default_viewport_cache_budget=32*mib,default_resource_backdrop_cache_budget=128*mib;
char const* option="";
unsigned GetEnvironmentVariableA(char const*,char* out,unsigned size){
    auto count=std::strlen(option);assert(count<size);std::strcpy(out,option);return unsigned(count);
}
struct State {
    std::size_t viewport_cache_budget=32*mib,resource_backdrop_cache_budget=128*mib,viewport_cache_bytes=1;
    std::vector<int> viewport_cache{1};unsigned releases=0;
    void clear_resource_backdrops(){++releases;}
    void configure(){char control[8]={};
''' + body + r'''}
};
int main(){
    State state;state.configure();assert(state.viewport_cache.size()==1 && !state.releases);
    option="1";state.configure();assert(state.viewport_cache.empty() && !state.viewport_cache_bytes && state.releases==1);
    assert(state.viewport_cache_budget==64*mib && state.resource_backdrop_cache_budget==832*mib);
    state.viewport_cache.push_back(1);state.viewport_cache_bytes=1;state.configure();
    assert(state.viewport_cache.size()==1 && state.releases==1);
    option="999999";state.configure(); // Unrecognized values cannot request unbounded storage.
    assert(state.viewport_cache.empty() && !state.viewport_cache_bytes && state.releases==2);
    assert(state.viewport_cache_budget==32*mib && state.resource_backdrop_cache_budget==128*mib);
}
'''
        run_cpp(program)

    def test_terrain_preview_anchors_ownership_and_input_bounds(self):
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
#include "Renderer/native/unit_animation_runtime.h"
#include "Renderer/native/render_core/unit_frame_preparation.h"
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
    textures[11].dds=texture(0xffff);textures[13].dds=texture(0xffff);
    tiles[0].real_terrain_type=11;assert(preview.render(frame,textures,out,cancelled));
    auto coast=out.pixels[16*128+32];assert((coast&255)>((coast>>16)&255));
    tiles[0].real_terrain_type=13;assert(preview.render(frame,textures,out,cancelled));
    auto ocean=out.pixels[16*128+32];assert((ocean&255)<(coast&255) && (ocean&255)>((ocean>>16)&255));
    tiles[0].real_terrain_type=2;
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
        run_cpp(program, sources=("Renderer/native/environment_runtime.cpp",))

    def test_camera_cancellation_reaches_draw_submission(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        submission = source.split('trace.write("geometry-ready", frame_cache_path);', 1)[1].split('trace.write("readback-begin"', 1)[0]
        self.assertIn("c3x_renderer::power_of_two_extent(height),foreground_pending,", submission)
        self.assertEqual(submission.count("if(cancelled())return false;"), 2)
        copy = source.split('trace.write("readback-begin"', 1)[1].split("bitmap_footprints =", 1)[0]
        self.assertLess(copy.index("if(cancelled())"), copy.index("cache_valid=false;"))
        self.assertLess(copy.index("cache_valid=false;"), copy.index("std::memmove"))

    def test_actual_worker_camera_supersession_and_takeover(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        publication = "struct PublishedMapFrame {" + source.split("struct PublishedMapFrame {", 1)[1].split("// Civ III remains", 1)[0]
        publication = publication.replace("auto first=static_cast<std::uint32_t const*>(source.bgra_pixels);",
                                          "publication_checkpoint(); auto first=static_cast<std::uint32_t const*>(source.bgra_pixels);")
        worker = "class RendererWorker {" + source.split("class RendererWorker {", 1)[1].split("RendererWorker * renderer_worker", 1)[0]
        worker = worker.replace("c3x_renderer::UnitBodyRenderer::PublishedPose", "Bodies::PublishedPose")
        worker = worker.replace("completed.wait(lock,[this,ticket]", "adoption_checkpoint(); completed.wait(lock,[this,ticket]")
        program = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
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
#include "Renderer/native/unit_animation_runtime.h"
#include "Renderer/native/render_core/unit_frame_preparation.h"
#include "Renderer/native/render_core/unit_playback.h"
#include "Renderer/native/render_core/cliff_placement.h"
#include "Renderer/native/prepared_view_area.h"
using HDC=void*;
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* out){out->QuadPart=std::chrono::steady_clock::now().time_since_epoch().count();}
bool ambient_mode=false,ahead_mode=false;
std::atomic<unsigned> ahead_completed{0},ahead_consumed{0};
unsigned GetEnvironmentVariableA(char const* name,char* out,std::size_t){
    if(!std::strcmp(name,"C3X_RENDERER_PREPARE_AHEAD")){std::strcpy(out,ahead_mode?"1":"0");return 1;}
    if(ambient_mode && !std::strcmp(name,"C3X_RENDERER_SYNC_AMBIENT")){std::strcpy(out,"1");return 1;}
    return 0;
}
#ifndef _MSC_VER
template<std::size_t N,class... T> void sprintf_s(char (&buffer)[N],char const* format,T... args){std::snprintf(buffer,N,format,args...);}
#endif
namespace c3x_renderer {
struct Signature {std::uint64_t complete=0;};
Signature terrain_frame_signature(c3x_renderer_frame_v1 const& f,long long,unsigned){
    return {std::uint64_t(f.presentation_time_ticks)+1};
}
}
std::atomic<unsigned> adoption_entered{0};
void adoption_checkpoint(){++adoption_entered;}
std::atomic<bool> hold_publication{false};
std::atomic<unsigned> publication_entered{0};
void publication_checkpoint(){++publication_entered;while(hold_publication.load())std::this_thread::yield();}
struct Trace {int level=0;bool buffered=false;void write(char const* stage,char const*,bool=false){if(!std::strcmp(stage,"ahead-prepared"))++ahead_completed;if(!std::strcmp(stage,"ahead-consumed"))++ahead_consumed;} double milliseconds(long long value){return double(value)/1000000;}};
using RendererTrace=Trace;
struct Footprint {int coordinate=0;struct {int left=0,right=0;} bounds;};
std::atomic<bool> hold_unit_pixels{false},unit_pixels_entered{false};
std::atomic<bool> check_demand_priority{false},demand_executed{false};
std::atomic<unsigned> priority_preparations{0};
struct Bodies {
    double payload_ms=0,pose_ms=0,submission_ms=0,readback_ms=0,output_ms=0;bool pose_content_hit=false;
    struct Stats {unsigned built=0,consumed=0,cancelled=0,evicted=0,rejected=0,active_peak=0;double cpu_ms=0,wait_ms=0;std::size_t bytes=0,peak_bytes=0;};
    Stats pose_preparation_statistics(){return {};}
    std::size_t pose_retained_bytes()const{return 0;}
    void release_pose_leases(){}

    struct Action {std::string name;bool loop=true,ambient=false;float duration=1;unsigned frames=31;};
    struct Unit {std::vector<std::string> keys;int minimum_canvas=0;std::vector<Action> actions;};
    struct PublishedPose {std::vector<unsigned> pixels;int width=191,height=191;unsigned cast_pixels=0;bool prepared=false;};
    bool copy_cached(c3x_renderer_unit_v1 const&,PublishedPose&){return cached;}
    bool blit(PublishedPose const&,HDC,int,int,HDC,unsigned&){return true;}
    std::size_t cached_pose_bytes()const{return cache_bytes;}
    template<class F> unsigned prepare_pixels(int,int,c3x_renderer_unit_v1 const* requests,unsigned count,F,std::atomic<bool> const& demanded){
        if(check_demand_priority && priority_preparations.fetch_add(1)>0)assert(demand_executed);
        assert(count<=2 && requests[0].action_cursor==1);unit_pixels_entered=true;
        while(hold_unit_pixels.load() && !demanded.load())std::this_thread::yield();return count;
    }
    std::vector<Unit> units={{{"unit"},0,{{c3x_renderer::native_unit_action(1),true}}}};int image_width=191,image_height=191;
    char const* failure_reason="";bool cache_hit=false,cached=false;std::size_t cache_bytes=0;unsigned keyed_pixels=0,cast_pixels=0;
    bool restore_cached(c3x_renderer_unit_v1 const&){cache_hit=cached;return cached;}
    std::size_t cached_pose_entries()const{return cached?1u:0u;}
    template<class F> bool render(int,int,c3x_renderer_unit_v1 const&,F,void* =nullptr,unsigned=1){demand_executed=true;return true;}
    bool blit(HDC,int,int,HDC){return true;}void reset_gpu(){}
};
struct RendererState {
    struct Terrain {bool configured=false;std::vector<std::uint8_t> dds;};
    std::array<Terrain,14> terrain_textures;
    Trace trace;Bodies unit_bodies;bool unit_rendering_enabled=true,pickup_profile=false,cache_valid=false,profiling=false;
    int device=0,context=0;
    unsigned cache_hits=0,device_recoveries=0,frame_tiles_built=0,prepared_blocks=0,visible_resource_animations=0;
    unsigned ambient_count() const {return visible_resource_animations;}
    bool can_prepare_ambient() const {return animate_pixels && visible_resource_animations;}
    std::size_t prefetched_geometry_bytes=0,tile_geometry_cache_bytes=0;
    std::uint64_t requested_signature=0;
    struct {std::size_t bytes=0;} pixel_blocks;
    struct {void clear(){}} geometry_cache;
    std::vector<Footprint> pixel_neighborhood;Footprint prepared_footprint;
    std::vector<c3x_renderer_tile_v1> cached_tiles;
    std::vector<unsigned> pixels,flags;
    std::atomic<unsigned> entered{0},cancelled{0},resets{0};
    std::atomic<bool> hold{false};
    std::atomic<long long> hold_clock{-1};
    std::atomic<unsigned> target_clock_entries{0};
    bool animate_pixels=false,fail_render=false,world_preparation=false,scene_guard_failed=false;
    bool shared_scene_surface=false;
    std::vector<std::uint64_t> prepared_view_dependencies()const{return {};}
    bool throw_cancellation=false,throw_failure=false;
    bool scene_guard_pending()const{return false;}
    bool prepare_scene_guard(std::atomic<bool> const&){return true;}
    bool render(c3x_renderer_frame_v1 const& f,c3x_renderer_output_v1& out,int=-1,
                std::atomic<bool> const* stop=nullptr,std::uint64_t=0,unsigned const* =nullptr,unsigned=0,c3x_renderer_frame_v1 const* =nullptr){
        ++entered;
        if(throw_failure)throw std::runtime_error("fixture runtime failure");
        if(f.presentation_time_ticks==101)++target_clock_entries;
        while(hold.load() || hold_clock.load()==f.presentation_time_ticks){
            if(stop && stop->load()){++cancelled;if(throw_cancellation)throw c3x_renderer::render_core::CliffPreparationCancelled{};return false;}
            std::this_thread::yield();
        }
        if(stop && stop->load()){++cancelled;if(throw_cancellation)throw c3x_renderer::render_core::CliffPreparationCancelled{};return false;}
        if(fail_render)return false;
        unsigned value=unsigned(f.tiles[0].anchor_x)^f.world_topology[0];
        if(animate_pixels)value^=unsigned(f.presentation_time_ticks);
        pixels.assign(std::size_t(f.target_width)*f.target_height,value);
        flags.clear();for(unsigned i=0;i<f.tile_count;++i)flags.push_back(f.tiles[i].tile_flags);
        cached_tiles.assign(f.tiles,f.tiles+f.tile_count);
        cache_valid=true;
        requested_signature=std::uint64_t(f.presentation_time_ticks)+1;
        out={C3X_RENDERER_API_VERSION,sizeof(out)};
        out.width=f.target_width;out.height=f.target_height;out.stride_bytes=out.width*4;
        out.bgra_pixels=pixels.data();out.replacement_tile_flags=flags.data();out.replacement_tile_count=f.tile_count;
        out.clip_right=out.width;out.clip_bottom=out.height;out.visible_animation_count=f.visible_animation_count+ambient_count();return true;
    }
    static long long resource_clock(c3x_renderer_frame_v1 const& f){return f.presentation_time_ticks;}
    bool configure_pack(char const*){reset();return true;}
    bool configure_definitions(char const*,char const*,char const*,char const*){reset();return true;}
    void reset(){demand_executed=true;++resets;pixels.clear();flags.clear();}
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
    c3x_renderer_frame_v1 f={};f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);f.target_width=2240;f.target_height=1192;f.tile_width=128;f.tile_height=64;
    f.tile_count=1;f.tiles=&tile;f.world_topology=&topology;f.world_topology_count=1;
    f.presentation_time_ticks=1;
    c3x_renderer_output_v1 out={C3X_RENDERER_API_VERSION,sizeof(out)};
    assert(worker.render(f,out)==C3X_RENDERER_RESULT_OK);
    c3x_renderer_i64 first=0,last=0;
    assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return worker.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
    c3x_renderer_i64 duplicate=0;
    auto published_entries=state.entered.load();
    assert(worker.camera_begin(f,duplicate)==C3X_RENDERER_RESULT_PENDING && duplicate==last);
    assert(worker.camera_poll(duplicate,out)==C3X_RENDERER_RESULT_OK && state.entered.load()==published_entries);
    auto old=out;auto old_value=7u^42u;
    state.hold=true;auto entered=state.entered.load();
    ++f.presentation_time_ticks;
    assert(worker.camera_begin(f,first)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    // Same content at different caller addresses keeps the active request.
    auto copy_tile=tile;auto copy_topology=topology;auto copy_frame=f;
    copy_frame.tiles=&copy_tile;copy_frame.world_topology=&copy_topology;
    for(int repeat=0;repeat<30;++repeat)
        assert(worker.camera_begin(copy_frame,duplicate)==C3X_RENDERER_RESULT_PENDING && duplicate==first);
    // Latest-wins under actual worker contention, not a detached queue model.
    for(unsigned i=0;i<30;++i){tile.anchor_x=int(100+i);topology=1000+i;++f.presentation_time_ticks;
        assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
        assert(worker.camera_begin(f,duplicate)==C3X_RENDERER_RESULT_PENDING && duplicate==last);}
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
    // A cached CPU-owned pose never interrupts the independent map worker.
    state.hold=true;entered=state.entered.load();++f.presentation_time_ticks;
    assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    auto cancellations=state.cancelled.load();
    c3x_renderer_unit_v1 unit={};unit.action=1;std::strcpy(unit.unit_key,"unit");state.unit_bodies.cached=true;
    assert(worker.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
    assert(state.cancelled.load()==cancellations && worker.camera_poll(last,out)==C3X_RENDERER_RESULT_PENDING);
    state.unit_bodies.cached=false;state.hold=false;
    until([&]{return worker.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
    // Unit takeover interrupts active work but preserves the latest immutable request.
    state.hold=true;entered=state.entered.load();++f.presentation_time_ticks;
    assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    assert(worker.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
    assert(worker.camera_poll(last,out)==C3X_RENDERER_RESULT_PENDING);
    assert(state.cache_valid); // Interruption before pixel mutation preserves the donor bitmap.
    // Repeated units neither grow a queue nor discard the requested map.
    for(int i=0;i<30;++i){
        assert(worker.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        assert(worker.camera_poll(last,out)==C3X_RENDERER_RESULT_PENDING);
    }
    state.hold=false;
    until([&]{return worker.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
    assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(tile.anchor_x)^topology));
    // Even rejected unit requests release the pause and resume the map.
    state.hold=true;entered=state.entered.load();
    ++f.presentation_time_ticks;
    assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    state.unit_rendering_enabled=false;
    assert(worker.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_ERROR);
    state.unit_rendering_enabled=true;state.hold=false;
    until([&]{return worker.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
    // Publish identity and exact ordered occurrences atomically with ownership.
    c3x_renderer_tile_v1 captured[2]={tile,tile};
    captured[0].anchor_x=31;captured[0].visibility_mask=4;
    captured[1].anchor_x=99;captured[1].tile_flags=8;captured[1].visibility_mask=8;
    f.tiles=captured;f.tile_count=2;f.world_topology_revision=73;
    c3x_renderer_camera_identity_v1 identity={11,22,33,44};
    c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
    view.ticket=-7;auto untouched=view;
    state.hold=true;
    assert(worker.camera_begin(f,last,identity)==C3X_RENDERER_RESULT_PENDING);
    assert(worker.camera_poll_view(last,view)==C3X_RENDERER_RESULT_PENDING);
    auto too_large=f;too_large.target_width=too_large.target_height=8192;
    c3x_renderer_i64 rejected_ticket=-1;
    assert(worker.camera_begin(too_large,rejected_ticket,identity)==C3X_RENDERER_RESULT_BAD_ARGUMENT);
    assert(rejected_ticket==-1 && worker.camera_poll_view(last,view)==C3X_RENDERER_RESULT_PENDING);
    assert(!std::memcmp(&view,&untouched,sizeof(view)));
    std::swap(captured[0],captured[1]);identity.visibility_epoch=34;
    state.hold=false;
    until([&]{return worker.camera_poll_view(last,view)==C3X_RENDERER_RESULT_OK;});
    assert(view.ticket==last && view.identity.map_epoch==11 && view.identity.viewer_epoch==22);
    assert(view.identity.visibility_epoch==33 && view.identity.scene_epoch==44);
    assert(view.frame.tiles!=captured && view.frame.tile_count==2);
    assert(view.frame.tiles[0].anchor_x==31 && view.frame.tiles[1].anchor_x==99);
    assert(view.frame.tiles[0].visibility_mask==4 && view.frame.tiles[1].visibility_mask==8);
    assert(view.output.replacement_tile_flags[0]==view.frame.tiles[0].tile_flags);
    assert(view.output.replacement_tile_flags[1]==view.frame.tiles[1].tile_flags);
    assert(!view.frame.world_topology && !view.frame.world_topology_count && view.frame.world_topology_revision==73);
    first=last;state.hold=true;
    assert(worker.camera_begin(f,last,identity)==C3X_RENDERER_RESULT_PENDING);
    untouched=view;
    assert(worker.camera_poll_view(first,view)==C3X_RENDERER_RESULT_SUPERSEDED);
    assert(!std::memcmp(&view,&untouched,sizeof(view)));
    assert(worker.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
    state.hold=false;
    until([&]{return worker.camera_poll_view(last,view)==C3X_RENDERER_RESULT_OK;});
    assert(view.identity.visibility_epoch==34 && view.frame.tiles[0].anchor_x==99);
    // The native caller must recapture its displayed view before acquiring old pixels.
    c3x_renderer_camera_request_v1 present={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(present),&f,identity};
    auto held=view;auto prior_clock=view.frame.presentation_time_ticks;
    f.presentation_time_ticks+=100;f.dirty_flags^=C3X_RENDERER_DIRTY_ALL;
    assert(worker.camera_present_view(present,held)==C3X_RENDERER_RESULT_OK);
    assert(held.frame.presentation_time_ticks==prior_clock && held.output.bgra_pixels==view.output.bgra_pixels);
    auto reject=[&](){auto before=held;assert(worker.camera_present_view(present,held)==C3X_RENDERER_RESULT_PENDING);
        assert(!std::memcmp(&before,&held,sizeof(held)));};
    ++present.identity.visibility_epoch;reject();--present.identity.visibility_epoch;
    ++present.identity.viewer_epoch;reject();--present.identity.viewer_epoch;
    ++present.identity.map_epoch;reject();--present.identity.map_epoch;
    ++present.identity.scene_epoch;reject();--present.identity.scene_epoch;
    ++f.world_topology_revision;reject();--f.world_topology_revision;
    ++f.target_width;reject();--f.target_width;
    ++f.tile_width;reject();--f.tile_width;
    ++captured[0].city_population;reject();--captured[0].city_population;
    ++captured[0].visibility_mask;reject();--captured[0].visibility_mask;
    std::swap(captured[0],captured[1]);reject();std::swap(captured[0],captured[1]);
    assert(worker.camera_present_view(present,held)==C3X_RENDERER_RESULT_OK);
    // Every identity/visibility/topology/time change still replaces the request.
    auto replace=[&](){
        auto previous=last;
        assert(worker.camera_begin(f,last,identity)==C3X_RENDERER_RESULT_PENDING && last>previous);
        assert(worker.camera_begin(f,duplicate,identity)==C3X_RENDERER_RESULT_PENDING && duplicate==last);
    };
    ++identity.map_epoch;replace();++identity.viewer_epoch;replace();++identity.visibility_epoch;replace();++identity.scene_epoch;replace();
    ++f.presentation_time_ticks;replace();++f.world_topology_revision;replace();
    ++topology;replace();++captured[0].visibility_mask;replace();
    std::swap(captured[0],captured[1]);replace();f.tile_width=160;f.tile_height=80;replace();
    f.tile_width=192;f.tile_height=96;replace();++f.dirty_flags;replace();
    until([&]{return worker.camera_poll_view(last,view)==C3X_RENDERER_RESULT_OK;});
    // A stalled completion copy cannot block identical begin, newer input or
    // polling; its obsolete output must never become the new publication.
    hold_publication=true;auto copying=publication_entered.load();++f.presentation_time_ticks;
    assert(worker.camera_begin(f,last,identity)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return publication_entered.load()>copying;});
    assert(worker.camera_begin(f,duplicate,identity)==C3X_RENDERER_RESULT_PENDING && duplicate==last);
    auto superseded=last;++identity.visibility_epoch;
    assert(worker.camera_begin(f,last,identity)==C3X_RENDERER_RESULT_PENDING && last>superseded);
    assert(worker.camera_poll(superseded,out)==C3X_RENDERER_RESULT_SUPERSEDED);
    assert(worker.camera_poll(last,out)==C3X_RENDERER_RESULT_PENDING);
    hold_publication=false;
    until([&]{return worker.camera_poll_view(last,view)==C3X_RENDERER_RESULT_OK;});
    assert(view.identity.visibility_epoch==identity.visibility_epoch && view.ticket==last);
    // Reset also joins an active cancellation, and never hangs on a pending job.
    state.hold=true;entered=state.entered.load();assert(worker.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
    until([&]{return state.entered.load()>entered;});
    worker.reset_and_stop();assert(state.resets==1);
    assert(worker.camera_poll(last,out)==C3X_RENDERER_RESULT_SUPERSEDED);
    // Legacy render and the explicit camera interface share one publication.
    // A matching active camera request cannot certify an older front's view.
    ambient_mode=true;
    {
        RendererState ambient;ambient.visible_resource_animations=1;ambient.animate_pixels=true;
        RendererWorker pull(ambient);
        f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);
        f.tiles=&tile;f.tile_count=1;
        f.presentation_time_ticks=1;tile.anchor_x=7;
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
        auto original_pixel=static_cast<unsigned const*>(out.bgra_pixels)[0];
        f.presentation_time_ticks=2;ambient.hold=true;
        auto before=ambient.entered.load();
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==original_pixel);
        until([&]{return ambient.entered.load()>before;});
        // Supersede the ambient job with an explicit different camera, while
        // keeping its render in flight and the old publication intact.
        tile.anchor_x=29;before=ambient.entered.load();
        assert(pull.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
        until([&]{return ambient.entered.load()>before;});
        before=ambient.entered.load();auto adopted=adoption_entered.load();std::atomic<bool> returned{false};
        std::thread caller([&]{assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);returned=true;});
        until([&]{return returned.load() || adoption_entered.load()>adopted;});
        assert(!returned.load()); // Must synchronously obtain the current view.
        ambient.hold=false;caller.join();
        assert(ambient.entered.load()==before); // Reuse the exact already-running job.
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(tile.anchor_x)^topology^2u));
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]!=original_pixel);
        // Same-view clock work remains passive until the next render call.
        f.presentation_time_ticks=3;before=ambient.entered.load();
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
        until([&]{return ambient.entered.load()>before;});
        // Calls may arrive while pending or after completion. Neither changes
        // the camera, and consuming this clock must not create another job.
        until([&]{assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
            return static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(tile.anchor_x)^topology^3u);});
        for(unsigned i=0;i<10;++i)assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
        assert(ambient.entered.load()==before+1);
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(tile.anchor_x)^topology^3u));
        c3x_renderer_camera_identity_v1 epochs={1,2,3,4};
        assert(pull.camera_begin(f,last,epochs)==C3X_RENDERER_RESULT_PENDING);
        until([&]{return pull.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
        before=ambient.entered.load();adopted=adoption_entered.load();
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
        assert(ambient.entered.load()==before+1 && adoption_entered.load()==adopted);
    }
    {
        RendererState ambient;ambient.visible_resource_animations=1;ambient.animate_pixels=true;
        RendererWorker pull(ambient);c3x_renderer_camera_identity_v1 epochs={21,22,23,24};
        f.tiles=&tile;f.tile_count=1;f.presentation_time_ticks=1;
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        auto previous=static_cast<unsigned const*>(out.bgra_pixels)[0];
        f.presentation_time_ticks=2;ambient.hold=true;auto before=ambient.entered.load();
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==previous);
        until([&]{return ambient.entered.load()>before;});
        // Visibility loss cannot return the older ambient front, even when
        // the camera, requested clock and payload bytes otherwise match.
        ++epochs.visibility_epoch;std::atomic<bool> returned{false};before=ambient.entered.load();
        std::thread native([&]{assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);returned=true;});
        until([&]{return ambient.entered.load()>before;});assert(!returned.load());
        ambient.hold=false;native.join();
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]!=previous);
        f.presentation_time_ticks=3;before=ambient.entered.load();
        until([&]{assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
            return static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(tile.anchor_x)^topology^3u);});
        assert(ambient.entered.load()==before+1);
        for(unsigned i=0;i<10;++i)assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        assert(ambient.entered.load()==before+1);
    }
    ambient_mode=false;
    {
        RendererState state;RendererWorker pull(state);
        f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);
        f.tiles=&tile;f.tile_count=1;f.presentation_time_ticks=100;
        // Adoption also works without the ambient compatibility mode.
        state.hold=true;assert(pull.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
        until([&]{return state.entered.load()==1;});
        auto adopted=adoption_entered.load();std::atomic<bool> returned{false};
        std::thread caller([&]{assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);returned=true;});
        until([&]{return adoption_entered.load()>adopted;});
        assert(!returned.load() && !state.cancelled.load());state.hold=false;caller.join();
        assert(state.entered.load()==1 && !state.cancelled.load());
        assert(out.replacement_tile_count==1 && out.replacement_tile_flags[0]==tile.tile_flags);
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(tile.anchor_x)^topology));
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK && state.entered.load()==1);
        // A completed explicit publication can be consumed without execution.
        ++f.presentation_time_ticks;
        assert(pull.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
        until([&]{return pull.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
        auto executions=state.entered.load();
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK && state.entered.load()==executions);
        // The exact request may still be pending behind an obsolete completion
        // copy. Joining must preserve it without a third snapshot or execution.
        hold_publication=true;auto copying=publication_entered.load();
        executions=state.entered.load();++f.presentation_time_ticks;
        assert(pull.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
        until([&]{return publication_entered.load()>copying;});
        ++f.presentation_time_ticks;
        assert(pull.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
        adopted=adoption_entered.load();returned=false;
        std::thread pending([&]{assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);returned=true;});
        until([&]{return adoption_entered.load()>adopted;});
        assert(!returned.load() && state.entered.load()==executions+1);
        hold_publication=false;pending.join();
        assert(state.entered.load()==executions+2);
        // Lifecycle identity is not available through legacy render. Even
        // byte-identical frame data cannot adopt an epoch-bearing request.
        for(unsigned which=0;which<4;++which){
            c3x_renderer_camera_identity_v1 epoch={};
            if(which==0)epoch.map_epoch=1;if(which==1)epoch.viewer_epoch=1;
            if(which==2)epoch.visibility_epoch=1;if(which==3)epoch.scene_epoch=1;
            ++f.presentation_time_ticks;
            assert(pull.camera_begin(f,last,epoch)==C3X_RENDERER_RESULT_PENDING);
            until([&]{return pull.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
            executions=state.entered.load();adopted=adoption_entered.load();
            assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
            assert(state.entered.load()==executions+1 && adoption_entered.load()==adopted);
        }
        // Native ordinary demand can join nonzero identities without another
        // render, then reject each independently changed lifecycle component.
        c3x_renderer_camera_identity_v1 epochs={11,12,13,14};
        for(unsigned which=0;which<4;++which){
            ++f.presentation_time_ticks;state.hold=true;
            executions=state.entered.load();
            assert(pull.camera_begin(f,last,epochs)==C3X_RENDERER_RESULT_PENDING);
            until([&]{return state.entered.load()==executions+1;});
            auto before_cancel=state.cancelled.load();adopted=adoption_entered.load();returned=false;
            std::thread native([&]{assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);returned=true;});
            until([&]{return adoption_entered.load()>adopted;});
            assert(!returned.load() && state.cancelled.load()==before_cancel);
            state.hold=false;native.join();assert(state.entered.load()==executions+1);
            assert(pull.camera_poll_view(last,view)==C3X_RENDERER_RESULT_OK);
            assert(!std::memcmp(&view.identity,&epochs,sizeof(epochs)));
            assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK && state.entered.load()==executions+1);
            if(which==0)++epochs.map_epoch;if(which==1)++epochs.viewer_epoch;
            if(which==2)++epochs.visibility_epoch;if(which==3)++epochs.scene_epoch;
            adopted=adoption_entered.load();
            assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
            assert(state.entered.load()==executions+2 && adoption_entered.load()==adopted);
            // The ordinary publication's identity must also guard its fast cache.
            assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK && state.entered.load()==executions+2);
        }
        // Distinct clock, camera, visibility, order and topology require their
        // own rendering even when an older result is ready.
        for(unsigned which=0;which<5;++which){
            c3x_renderer_tile_v1 pair[2]={tile,tile};pair[1].anchor_x+=100;
            auto current=f;current.tiles=pair;current.tile_count=2;
            ++current.presentation_time_ticks;
            assert(pull.camera_begin(current,last)==C3X_RENDERER_RESULT_PENDING);
            until([&]{return pull.camera_poll(last,out)==C3X_RENDERER_RESULT_OK;});
            executions=state.entered.load();adopted=adoption_entered.load();
            if(which==0)++current.presentation_time_ticks;
            if(which==1)++pair[0].anchor_x;
            if(which==2)++pair[0].visibility_mask;
            if(which==3)std::swap(pair[0],pair[1]);
            if(which==4)++topology;
            assert(pull.render(current,out)==C3X_RENDERER_RESULT_OK);
            assert(state.entered.load()==executions+1 && adoption_entered.load()==adopted);
            assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(pair[0].anchor_x)^topology));
        }
        ++f.presentation_time_ticks;state.hold=true;state.fail_render=true;
        auto before=state.entered.load();
        assert(pull.camera_begin(f,last)==C3X_RENDERER_RESULT_PENDING);
        until([&]{return state.entered.load()>before;});
        adopted=adoption_entered.load();returned=false;auto untouched=out;
        std::thread failed([&]{assert(pull.render(f,out)==C3X_RENDERER_RESULT_DEVICE_ERROR);returned=true;});
        until([&]{return adoption_entered.load()>adopted;});
        assert(!returned.load());state.hold=false;failed.join();
        assert(!std::memcmp(&out,&untouched,sizeof(out))); // No old front reported as success.
        state.fail_render=false;
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
        pull.reset_and_stop();assert(pull.camera_poll(last,out)==C3X_RENDERER_RESULT_SUPERSEDED);
    }

    ahead_mode=true;
    {
        RendererState state;state.visible_resource_animations=1;state.animate_pixels=true;
        RendererWorker pull(state);c3x_renderer_camera_identity_v1 epochs={3,4,5,6};
        f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);
        f.tiles=&tile;f.tile_count=1;f.presentation_frequency=15;f.presentation_time_ticks=19;
        f.target_width=64;f.target_height=32;tile.anchor_x=7;topology=42;
        auto prepared=ahead_completed.load();auto consumed=ahead_consumed.load();
        auto copies=publication_entered.load();
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        assert(publication_entered.load()==copies);
        // A changed view does not compete with scrolling for future GPU work.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));assert(ahead_completed.load()==prepared);
        f.presentation_time_ticks=20;
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        auto first=out;auto original=static_cast<unsigned const*>(out.bgra_pixels)[0];
        assert(publication_entered.load()>copies);
        until([&]{return ahead_completed.load()>=prepared+2;});
        assert(state.entered.load()==4); // Finite horizon; no autonomous clock loop.
        assert(static_cast<unsigned const*>(first.bgra_pixels)[0]==original);
        for(unsigned i=0;i<20;++i)assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        assert(state.entered.load()==4);
        // Caller buffers can change address without changing their identity.
        auto copy=tile;auto world=topology;f.tiles=&copy;f.world_topology=&world;
        f.presentation_time_ticks=21;
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        assert(ahead_consumed.load()==consumed+1);
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(7u^42u^21u));
        assert(out.replacement_tile_flags[0]==copy.tile_flags && !out.geometry_tiles_built);
        // Consume the exact second slot even if the new horizon is in flight.
        f.presentation_time_ticks=22;
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(7u^42u^22u));
        // Changed appearance/topology/epochs reject ready work, even with the
        // same nominal topology revision. All four native epochs are checked.
        auto reject=[&](){
            auto hits=ahead_consumed.load();
            assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
            assert(ahead_consumed.load()==hits);
            assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(copy.anchor_x)^world^unsigned(f.presentation_time_ticks)));
        };
        ++world;++f.presentation_time_ticks;reject();
        ++copy.visibility_mask;++f.presentation_time_ticks;reject();
        ++copy.anchor_x;++f.presentation_time_ticks;reject();
        ++epochs.visibility_epoch;++f.presentation_time_ticks;reject();
        ++epochs.scene_epoch;++f.presentation_time_ticks;reject();
        ++epochs.viewer_epoch;++f.presentation_time_ticks;reject();
        ++epochs.map_epoch;++f.presentation_time_ticks;reject();
        copies=publication_entered.load();
        ++copy.anchor_x;++f.presentation_time_ticks;reject();
        assert(publication_entered.load()==copies); // Camera takeover borrows exact synchronous output.
        --f.presentation_time_ticks;reject(); // Clock reversal is never a future hit.
        // An active matching prediction is joined, not executed twice.
        pull.reset_and_stop();state.hold=false;
        f.presentation_time_ticks=99;prepared=ahead_completed.load();
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        f.presentation_time_ticks=100;
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        state.hold=true;
        auto entered=state.entered.load();
        until([&]{return state.entered.load()>entered;});
        f.presentation_time_ticks=101;std::atomic<bool> returned{false};
        auto before=state.entered.load();
        std::thread caller([&]{assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);returned=true;});
        state.hold=false;caller.join();
        assert(returned && static_cast<unsigned const*>(out.bgra_pixels)[0]==(unsigned(copy.anchor_x)^world^101u));
        assert(state.entered.load()<=before+1); // At most the next speculative tick.
        // Uncached unit takeover and reset retire work without hanging or
        // freeing the immutable front while a caller can still blit it.
        state.hold=true;before=state.entered.load();
        until([&]{return state.entered.load()>before || ahead_completed.load()>=prepared+2;});
        state.unit_bodies.cached=false;
        assert(pull.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        auto saved=static_cast<unsigned const*>(out.bgra_pixels)[0];
        assert(pull.blit(out,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        assert(static_cast<unsigned const*>(out.bgra_pixels)[0]==saved);
        pull.reset_and_stop();
    }
    // Native camera requests take ownership of an already-running ambient
    // result. The caller remains nonblocking and the exact bucket renders once.
    {
        RendererState state;state.visible_resource_animations=1;state.animate_pixels=true;
        RendererWorker pull(state);c3x_renderer_camera_identity_v1 epochs={3,4,5,6};
        f.tiles=&tile;f.tile_count=1;f.world_topology=&topology;f.world_topology_count=1;
        f.presentation_frequency=15;f.presentation_time_ticks=99;
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&f,epochs};
        assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_PENDING);
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        state.hold_clock=101;f.presentation_time_ticks=100;
        assert(pull.render(f,out,epochs)==C3X_RENDERER_RESULT_OK);
        until([&]{return state.target_clock_entries.load()==1;});
        auto cancellations=state.cancelled.load();auto hits=ahead_consumed.load();
        f.presentation_time_ticks=101;
        assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_OK);
        assert(view.frame.presentation_time_ticks==100); // No wait for held producer.
        assert(pull.camera_begin(f,last,epochs)==C3X_RENDERER_RESULT_PENDING);
        assert(pull.camera_poll_view(last,view)==C3X_RENDERER_RESULT_PENDING);
        assert(state.cancelled.load()==cancellations);
        state.hold_clock=-1;
        until([&]{return pull.camera_poll_view(last,view)==C3X_RENDERER_RESULT_OK;});
        assert(state.target_clock_entries.load()==1 && state.cancelled.load()==cancellations);
        assert(ahead_consumed.load()==hits+1);
        assert(view.frame.presentation_time_ticks==101);
        assert(static_cast<unsigned const*>(view.output.bgra_pixels)[0]==(unsigned(tile.anchor_x)^topology^101u));
        assert(!view.output.renderer_cpu_ticks);
        ++tile.visibility_mask;
        assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_PENDING);
        pull.reset_and_stop();
    }
    ahead_mode=false;
    {
        RendererState state;RendererWorker pull(state);
        // Use the actual native action name and hold the GPU preparation owner.
        state.unit_bodies.units[0].actions[0].name=c3x_renderer::native_unit_action(1);
        state.unit_bodies.cached=true;unit.action=1;unit.action_cursor=0;unit.frame_count=15;
        std::strcpy(unit.unit_key,"unit");hold_unit_pixels=true;unit_pixels_entered=false;
        assert(pull.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        until([&]{return unit_pixels_entered.load();});
        // Current cached body copies complete while the GPU preparation remains held.
        for(int i=0;i<20;++i)assert(pull.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        assert(hold_unit_pixels.load());hold_unit_pixels=false;pull.reset_and_stop();
    }
    // A completed surrounding area serves the actual new camera without
    // entering D3D; the next producer owns copied input and can be cancelled.
    {
        RendererState state;RendererWorker pull(state);state.shared_scene_surface=true;
        state.visible_resource_animations=1;state.animate_pixels=true;
        auto area_frame=f;auto area_tile=tile;area_tile.tile_x=area_tile.tile_y=20;
        area_tile.anchor_x=area_tile.anchor_y=0;area_tile.tile_flags=C3X_RENDERER_TILE_RENDER;
        area_frame.tiles=&area_tile;area_frame.target_width=128;area_frame.target_height=96;
        area_frame.presentation_frequency=1000;area_frame.presentation_time_ticks=100;
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&area_frame,{1,2,3,4}};
        c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
        assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_PENDING);
        assert(pull.render(area_frame,out,request.identity)==C3X_RENDERER_RESULT_OK);
        assert(pull.prepare_nearby_view(request)==C3X_RENDERER_RESULT_OK);
        area_tile.anchor_x=-32;
        until([&]{return pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_OK;});
        auto entries=state.entered.load();
        for(int i=0;i<5;++i)assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_OK);
        assert(state.entered.load()==entries && view.frame.tiles[0].anchor_x==-32);
        ++area_frame.presentation_time_ticks;
        assert(pull.prepare_nearby_view(request)==C3X_RENDERER_RESULT_OK);
        until([&]{return pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_OK && view.frame.presentation_time_ticks==101;});
        // Refresh keeps the area's original world placement, independent of
        // the current camera's -32 pixel occurrence translation.
        assert(static_cast<unsigned const*>(view.output.bgra_pixels)[0]==(128u^topology^101u));
        entries=state.entered.load();
        ++area_tile.visibility_mask;
        assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_PENDING);
        state.hold=true;
        assert(pull.prepare_nearby_view(request)==C3X_RENDERER_RESULT_OK);
        until([&]{return state.entered.load()>entries;});
        area_tile.anchor_x=-64; // active job already owns the earlier snapshot
        assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_PENDING);
        pull.reset_and_stop();assert(state.cancelled.load()>0);
        assert(pull.camera_present_view(request,view)==C3X_RENDERER_RESULT_PENDING);
    }
    // Recursive source compilation can unwind on supersession. That must
    // retire the partial assembly without reloading assets or world content.
    for(bool failure:{false,true}) {
        RendererState state;RendererWorker pull(state);
        state.throw_cancellation=!failure;state.throw_failure=failure;
        state.hold=!failure;
        c3x_renderer_i64 ticket=0;
        assert(pull.camera_begin(f,ticket)==C3X_RENDERER_RESULT_PENDING);
        until([&]{return state.entered.load()>0;});
        pull.camera_cancel(ticket);
        until([&]{return failure?state.resets.load()>0:state.cancelled.load()>0;});
        state.hold=false;state.throw_failure=false;
        assert(pull.render(f,out)==C3X_RENDERER_RESULT_OK);
        assert(state.resets.load()==unsigned(failure));
        pull.reset_and_stop();
    }
    // A waiting native draw or configuration/reset owns the next GPU turn.
    // Finishing one optional job must not immediately start another while the
    // caller is still reacquiring the queue mutex after its condition wait.
    for(bool reset:{false,true}) {
        RendererState state;RendererWorker pull(state);
        state.unit_bodies.cached=true;unit.unit_id=10;unit.direction=1;
        unit.action=1;unit.action_cursor=0;unit.frame_count=15;
        hold_unit_pixels=true;unit_pixels_entered=false;priority_preparations=0;
        check_demand_priority=true;demand_executed=false;
        assert(pull.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        until([&]{return unit_pixels_entered.load();});
        auto other=unit;other.unit_id=11;
        assert(pull.draw_unit(other,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        // Keep a second prediction pending while the first waits for demand.
        if(reset)pull.reset_and_stop();
        else {
            state.unit_bodies.cached=false;unit.direction=2;
            assert(pull.draw_unit(unit,reinterpret_cast<HDC>(1))==C3X_RENDERER_RESULT_OK);
        }
        assert(demand_executed);hold_unit_pixels=false;pull.reset_and_stop();
        check_demand_priority=false;
    }
}
'''
        run_cpp(program, sources=("Renderer/native/environment_runtime.cpp",))

    def test_ui_blitter_resource_lifetime_and_failures(self):
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
        run_cpp(program)

    def test_owned_output_atomic_failure_and_budget(self):
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
    c3x_renderer_tile_v1 tiles[2]={};tiles[0].anchor_x=123;tiles[1].visibility_mask=8;
    c3x_renderer_frame_v1 frame={};frame.target_width=2240;frame.target_height=1192;frame.tiles=tiles;frame.tile_count=2;
    frame.world_topology_revision=79;
    out.replacement_tile_count=2;out.replacement_tile_flags=flags;
    c3x_renderer_camera_identity_v1 identity={1,2,3,4};
    assert(published.capture(out,4,5,&frame,identity));
    assert(published.frame.tiles!=tiles && published.frame.tiles[0].anchor_x==123);
    assert(published.frame.tiles[1].visibility_mask==8 && published.identity.viewer_epoch==2);
    assert(published.frame.world_topology_revision==79);
    old_pixels=published.output.bgra_pixels;
    auto old_occurrences=published.frame.tiles;
    for(int failure=0;failure<3;++failure){
        allocations_before_failure=failure;
        assert(!published.capture(out,0,0,&frame,{}));allocations_before_failure=-1;
        assert(published.output.bgra_pixels==old_pixels && published.frame.tiles==old_occurrences);
        assert(published.identity.visibility_epoch==3);
    }
    frame.tile_count=3;assert(!published.capture(out,0,0,&frame,{}));frame.tile_count=2;
    frame.target_width=17;assert(!published.capture(out,0,0,&frame,{}));frame.target_width=2240;
    unsigned invalid_index=2;out.fallback_tile_count=1;out.fallback_tile_indices=&invalid_index;
    assert(!published.capture(out,0,0,&frame,{}));
    assert(published.frame.tiles==old_occurrences && published.identity.visibility_epoch==3);
    published.clear();assert(!published.output.bgra_pixels && published.pixels.capacity()==0);
    assert(published.fallback.capacity()==0 && published.replacements.capacity()==0 && published.occurrences.capacity()==0);
}
'''
        run_cpp(program)


if __name__ == "__main__":
    unittest.main()
