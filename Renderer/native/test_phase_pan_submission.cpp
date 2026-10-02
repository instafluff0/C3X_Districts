// Windows-only, untimed regression client for actual production fresh-scene exports.
// From an x64 Visual Studio developer prompt at the repository root, first create
// Renderer\.cache\phase-pan-submission, then compile into that disposable folder:
// cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 /DC3X_SANDBOX_CLIENT /I. /IRenderer\native Renderer\sandbox\reference_x64.cpp Renderer\native\test_phase_pan_submission.cpp /Fo:Renderer\.cache\phase-pan-submission\ /Fe:Renderer\.cache\phase-pan-submission\phase_pan_oracle.exe /link gdi32.lib user32.lib
// CLI: phase_pan_oracle.exe DLL MOD_ROOT DEFINITIONS SCENE.csv INITIAL.bmp WIDTH HEIGHT CENTER_X CENTER_Y TILE_WIDTH HOUR
// Use a fresh existing C3X_PHASE_PAN_CAPTURE_DIR; retain all captures on failure.
// Require C3X_SANDBOX_UNITS=0, C3X_SANDBOX_WHOLE_WORLD=1, C3X_SANDBOX_SHADOW_PATCHES=1,
// C3X_RENDERER_SHARED_SCENE_SURFACE=1, C3X_RENDERER_WATER_MOTION=1, C3X_RENDERER_WAVES=1,
// C3X_RENDERER_PREVIEW_ANIMATION=1 and the normal shader source root. Clear inherited
// preview session/replay/ablation and water/wave/reflection diagnostic skip options.
// Scope: exact BGRA/raw depth for pan, clock, zoom, wrap, forced raster, history and
// full reset with units disabled and normal effects. Draw/capture pairs execute on
// one queued render-owner transaction. Untimed readbacks are test-only;
// this does not measure FPS or certify gameplay, native HUD, scanout or positive
// authored wave phases (those use the separate host constant-byte oracle).

// Private untimed client of the existing production fresh-scene exports.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/gpu_frame_api.h"
#include "Renderer/sandbox/capture_model.h"

int sandbox_reference_main(int,char**);
char const* oracle_mod_root=nullptr;
char const* oracle_definitions=nullptr;

void oracle_require(bool ok,char const* name){if(!ok)throw std::runtime_error(name);}
std::vector<unsigned char> oracle_bytes(std::string const& path){
    FILE* file=nullptr;oracle_require(!fopen_s(&file,path.c_str(),"rb")&&file,"capture open");
    oracle_require(!fseek(file,0,SEEK_END),"capture seek");long count=ftell(file);
    oracle_require(count>0&&!fseek(file,0,SEEK_SET),"capture extent");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(count));
    bool ok=fread(bytes.data(),1,bytes.size(),file)==bytes.size();ok=!fclose(file)&&ok;
    oracle_require(ok,"capture read");return bytes;
}

using OracleDraw=int(*)(c3x_renderer_frame_v1 const*,char const*,int,int,int,int,int,int,int,float);
struct OracleWitness {
    c3x_renderer_frame_v1 const* frame;
    OracleDraw draw;
    int(*capture)(char const*);
    void(*invalidate)();
    int native_x,native_y;float zoom;
    char const* prefix;char const* forced_prefix;
    DWORD caller_thread,owner_thread=0;
    char error[192]{};
};
// Only direct scene/test exports run here. Camera adoption, reset, CPU byte
// comparisons and trace flushing stay outside this worker-owned transaction.
int oracle_witness(void* context){
    auto& job=*static_cast<OracleWitness*>(context);job.owner_thread=GetCurrentThreadId();
    try{
        oracle_require(job.owner_thread!=job.caller_thread,"witness must run on render owner");
        oracle_require(!job.draw(job.frame,nullptr,job.native_x,job.native_y,24,56,1,47,0,job.zoom),"fresh draw");
        oracle_require(!job.capture(job.prefix),"GPU capture");
        if(job.forced_prefix){
            job.invalidate();
            oracle_require(!job.draw(job.frame,nullptr,job.native_x,job.native_y,24,56,1,47,0,job.zoom),"forced raster draw");
            oracle_require(!job.capture(job.forced_prefix),"forced GPU capture");
        }
        return 0;
    }catch(std::exception const& error){std::snprintf(job.error,sizeof(job.error),"%s",error.what());}
    catch(...){std::snprintf(job.error,sizeof(job.error),"unknown witness exception");}
    return 1;
}
struct OracleCaptures {std::string prefix,forced;};

int sandbox_client_run(HMODULE module,c3x_renderer_frame_v1 const& original){
    try{
        using Begin=int(*)(c3x_renderer_camera_request_v1 const*,c3x_renderer_i64*);
        using Poll=int(*)(c3x_renderer_i64,c3x_renderer_gpu_camera_view_v1*);
        using Untimed=int(*)(int(*)(void*),void*);
        auto begin=reinterpret_cast<Begin>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
        auto ready=reinterpret_cast<Poll>(GetProcAddress(module,"c3x_renderer_trial_camera_ready"));
        auto adopt=reinterpret_cast<Poll>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll_view"));
        auto draw=reinterpret_cast<OracleDraw>(GetProcAddress(module,"c3x_sandbox_draw_fresh"));
        auto capture=reinterpret_cast<int(*)(char const*)>(GetProcAddress(module,"c3x_sandbox_witness_capture"));
        auto invalidate=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_sandbox_scroll_cache_invalidate"));
        auto untimed=reinterpret_cast<Untimed>(GetProcAddress(module,"c3x_renderer_trial_untimed_witness"));
        auto flush=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_trial_trace_flush"));
        auto reset=reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(module,"c3x_renderer_reset"));
        auto definitions=reinterpret_cast<c3x_renderer_set_definition_paths_fn>(GetProcAddress(module,"c3x_renderer_set_definition_paths"));
        oracle_require(begin&&ready&&adopt&&draw&&capture&&invalidate&&untimed&&flush&&reset&&definitions,"production exports");
        char folder[4*MAX_PATH]{};
        oracle_require(GetEnvironmentVariableA("C3X_PHASE_PAN_CAPTURE_DIR",folder,sizeof(folder))!=0,"capture folder");
        SandboxCameraWitness witness(original);
        c3x_renderer_camera_identity_v1 identity={1,2,3,4};
        struct Sample {SandboxCameraWitness::View view;c3x_renderer_i64 tick;};
        int span=original.world_width_tiles*original.tile_width/2;
        std::vector<Sample> samples={
            {{0,0,1,"origin"},29500},{{16,8,1,"pan16_8"},29500},
            {{32,16,1,"pan32_16"},29500},{{16,8,1,"return16_8"},29500},
            {{0,0,1,"origin_return"},29500},{{0,0,1,"clock_advance"},29533},
            {{0,0,1,"clock_zero"},0},{{0,0,1,"clock_return"},29500},
            {{0,0,1.5f,"zoom192"},29500},{{0,0,1,"zoom_return"},29500},
            {{span+16,8,1,"wrap16_8"},29500}};
        c3x_renderer_frame_v1 selected_frame={};
        std::vector<c3x_renderer_tile_v1> selected_tiles;
        std::vector<c3x_renderer_u32> selected_topology;
        bool selected_valid=false;
        auto select=[&](c3x_renderer_frame_v1 const& frame){
            // Projection-only changes use the already adopted immutable source.
            // The ready queue is consumed at adoption, so an identical begin
            // intentionally has no second completion to wait for.
            auto metadata=frame;metadata.tiles=nullptr;metadata.world_topology=nullptr;
            bool same=selected_valid&&!std::memcmp(&metadata,&selected_frame,sizeof(metadata))&&
                selected_tiles.size()==frame.tile_count&&
                !std::memcmp(selected_tiles.data(),frame.tiles,sizeof(frame.tiles[0])*frame.tile_count)&&
                selected_topology.size()==frame.world_topology_count&&
                (!frame.world_topology_count||!std::memcmp(selected_topology.data(),frame.world_topology,
                    sizeof(frame.world_topology[0])*frame.world_topology_count));
            if(same)return;
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,identity};
            c3x_renderer_i64 ticket=0;
            oracle_require(begin(&request,&ticket)==C3X_RENDERER_RESULT_PENDING,"camera begin");
            c3x_renderer_gpu_camera_view_v1 shown={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(shown)};
            ULONGLONG deadline=GetTickCount64()+120000;int result;
            do{result=ready(ticket,&shown);if(result==C3X_RENDERER_RESULT_PENDING)Sleep(1);}
            while(result==C3X_RENDERER_RESULT_PENDING&&GetTickCount64()<deadline);
            oracle_require(result==C3X_RENDERER_RESULT_OK,"camera ready");
            do{result=adopt(ticket,&shown);if(result==C3X_RENDERER_RESULT_PENDING)Sleep(1);}
            while(result==C3X_RENDERER_RESULT_PENDING&&GetTickCount64()<deadline);
            oracle_require(result==C3X_RENDERER_RESULT_OK,"camera adopt");
            auto const& actual=shown.camera.frame;
            oracle_require(shown.camera.ticket==ticket&&actual.tile_count==frame.tile_count&&
                actual.presentation_time_ticks==frame.presentation_time_ticks&&actual.tile_width==frame.tile_width&&
                actual.target_width==frame.target_width&&actual.target_height==frame.target_height&&
                !std::memcmp(&shown.camera.identity,&identity,sizeof(identity))&&actual.tiles&&
                !std::memcmp(actual.tiles,frame.tiles,sizeof(frame.tiles[0])*frame.tile_count),"exact selected source");
            selected_frame=metadata;selected_tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
            selected_topology.clear();if(frame.world_topology_count)
                selected_topology.assign(frame.world_topology,frame.world_topology+frame.world_topology_count);
            selected_valid=true;
        };
        DWORD witness_owner=0;unsigned serialized_bundles=0;
        auto save=[&](c3x_renderer_frame_v1 const& frame,int native_x,int native_y,float zoom,char const* name,bool forced){
            OracleCaptures files{std::string(folder)+"\\"+name,{}};
            if(forced)files.forced=files.prefix+"_forced_raster";
            OracleWitness job{&frame,draw,capture,invalidate,native_x,native_y,zoom,
                files.prefix.c_str(),forced?files.forced.c_str():nullptr,GetCurrentThreadId()};
            int result=untimed(oracle_witness,&job);
            oracle_require(result==C3X_RENDERER_RESULT_OK,job.error[0]?job.error:"queued witness failed");
            oracle_require(job.owner_thread && job.owner_thread!=job.caller_thread,"owned witness callback");
            oracle_require(!witness_owner || witness_owner==job.owner_thread,"one render owner per worker lifetime");
            witness_owner=job.owner_thread;++serialized_bundles;
            std::printf("PHASE_PAN_OWNER owner_thread=%lu caller_thread=%lu serialized=1 forced=%u\n",
                static_cast<unsigned long>(job.owner_thread),static_cast<unsigned long>(job.caller_thread),unsigned(forced));
            return files;
        };
        auto same=[&](std::string const& a,std::string const& b){
            return oracle_bytes(a+".bmp")==oracle_bytes(b+".bmp")&&oracle_bytes(a+".depth")==oracle_bytes(b+".depth");};
        unsigned raster_pairs=0,history_pairs=0,mismatched_pairs=0;
        for(auto const& sample:samples){
            auto tiles=witness.capture(sample.view);oracle_require(!tiles.empty(),"copied capture");
            auto frame=original;frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            frame.presentation_frequency=1000;frame.presentation_time_ticks=sample.tick;
            frame.dirty_flags=C3X_RENDERER_DIRTY_ALL;
            select(frame);
            // Match the production callback's authoritative captured basis.
            // Relative route offsets are not the native camera coordinates.
            int native_x=frame.tiles[0].anchor_x-frame.tiles[0].tile_x*frame.tile_width/2;
            int native_y=frame.tiles[0].anchor_y-frame.tiles[0].tile_y*frame.tile_height/2;
            bool forced=!std::strcmp(sample.view.name,"origin")||!std::strcmp(sample.view.name,"pan16_8");
            auto files=save(frame,native_x,native_y,sample.view.zoom,sample.view.name,forced);
            auto const& prefix=files.prefix;
            if(forced){
                bool exact=same(prefix,files.forced);
                std::printf("PHASE_PAN_RASTER view=%s exact=%u\n",sample.view.name,unsigned(exact));
                if(!exact)++mismatched_pairs;++raster_pairs;
            }
            char const* previous=nullptr;
            if(!std::strcmp(sample.view.name,"return16_8"))previous="pan16_8";
            if(!std::strcmp(sample.view.name,"origin_return")||!std::strcmp(sample.view.name,"clock_return")||
                !std::strcmp(sample.view.name,"zoom_return"))previous="origin";
            if(previous){bool exact=same(std::string(folder)+"\\"+previous,prefix);
                std::printf("PHASE_PAN_HISTORY view=%s previous=%s exact=%u\n",sample.view.name,previous,unsigned(exact));
                std::fflush(stdout);if(!exact)++mismatched_pairs;++history_pairs;}
            std::printf("PHASE_PAN_SAMPLE view=%s tick=%lld camera=%d,%d zoom=%.3f exact_source=1 units=0 normal_effects=1\n",
                sample.view.name,static_cast<long long>(sample.tick),sample.view.x,sample.view.y,sample.view.zoom);
            std::fflush(stdout);flush();
        }
        // A new renderer/device/owner state prepares the same small-pan source.
        // The original copied input remains caller-owned during the reset.
        reset();selected_valid=false;witness_owner=0;char custom[4*MAX_PATH]{};GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS",custom,sizeof(custom));
        oracle_require(definitions(oracle_mod_root,oracle_definitions,nullptr,custom[0]?custom:nullptr)==C3X_RENDERER_RESULT_OK,"cold definitions");
        auto view=samples[1].view;auto tiles=witness.capture(view);auto frame=original;
        frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());frame.presentation_frequency=1000;
        frame.presentation_time_ticks=samples[1].tick;frame.dirty_flags=C3X_RENDERER_DIRTY_ALL;
        select(frame);
        int native_x=frame.tiles[0].anchor_x-frame.tiles[0].tile_x*frame.tile_width/2;
        int native_y=frame.tiles[0].anchor_y-frame.tiles[0].tile_y*frame.tile_height/2;
        auto files=save(frame,native_x,native_y,view.zoom,"pan16_8_reset",false);
        bool exact=same(std::string(folder)+"\\pan16_8",files.prefix);
        std::printf("PHASE_PAN_RESET view=pan16_8 exact=%u\n",unsigned(exact));
        if(!exact)++mismatched_pairs;flush();
        std::printf("PHASE_PAN_SUMMARY mismatched_pairs=%u strict_exact=%u serialized_bundles=%u\n",mismatched_pairs,unsigned(!mismatched_pairs),serialized_bundles);
        std::fflush(stdout);oracle_require(!mismatched_pairs,"complete color and depth comparison; inspect all preserved mismatches");
        std::printf("PASS phase pan oracle samples=%zu raster_pairs=%u reset_pairs=1 history_pairs=%u gpu_color_depth_exact=1 untimed=1 authored_positive_phase=host_byte_oracle_only\n",samples.size(),raster_pairs,history_pairs);
        return 0;
    }catch(std::exception const& error){
        auto flush_failed=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_trial_trace_flush"));
        if(flush_failed)flush_failed();
        std::fprintf(stderr,"PHASE_PAN_ORACLE_FAILED %s\n",error.what());return 1;
    }
}

int main(int argc,char** argv){
    if(argc!=12)return 2;oracle_mod_root=argv[2];oracle_definitions=argv[3];
    return sandbox_reference_main(argc,argv);
}
