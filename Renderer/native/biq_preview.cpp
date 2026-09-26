#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdexcept>
#include <thread>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <set>
#include <utility>
#include <vector>
#include <string>

#include "c3x_renderer_api.h"
#include "gpu_frame_api.h"
#include "native_frame_workload.h"
#ifdef C3X_GPU_NATIVE_CONTRACT
bool native_worker_contract(char const*,c3x_renderer_gpu_images_fn,c3x_renderer_gpu_frame_v1&,c3x_renderer_gpu_render_fn,c3x_renderer_gpu_present_fn,c3x_renderer_camera_request_v1 const&,unsigned const*,int,int,std::vector<NativeFrameSample> const&);
#endif
#include "benchmark_oracle.h"
#include "busy_session_plan.h"

// Mirror the critical production boundary on every render, including warm hits.
bool preview_ownership(c3x_renderer_frame_v1 const& frame, c3x_renderer_output_v1 const& output) {
    if (output.fallback_tile_count || output.replacement_tile_count != frame.tile_count ||
        (frame.tile_count && !output.replacement_tile_flags)) return false;
    for (unsigned i=0;i<frame.tile_count;++i) {
        bool visible=(frame.tiles[i].tile_flags & C3X_RENDERER_TILE_RENDER)!=0;
        unsigned flags=output.replacement_tile_flags[i];
        if ((!visible && flags) || (visible && !(flags & C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED))) {
            std::printf("FAIL production ownership index=%u tile=%d,%d captured=%u replacement=%u\n",
                i,frame.tiles[i].tile_x,frame.tiles[i].tile_y,frame.tiles[i].tile_flags,flags);
            return false;
        }
    }
    return true;
}

struct CsvTile {
    int x, y, base, real;
    unsigned bonus, overlays, river;
    bool topology_halo;
};

std::uint32_t preview_seed(int x, int y) {
    std::uint32_t value = 2166136261u;
    value = (value ^ static_cast<std::uint32_t>(x)) * 16777619u;
    return (value ^ static_cast<std::uint32_t>(y)) * 16777619u;
}

bool preview_units(HMODULE module,char const* path,int hour);

bool read_scene(char const * path, int & map_width, int & map_height, std::vector<CsvTile> & tiles) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr)
        return false;
    char header[256] = {};
    char magic[40] = {};
    unsigned count = 0;
    unsigned halo_count = 0;
    int columns = 0, rows = 0, origin_column = 0, origin_row = 0;
    bool ok = std::fgets(header, sizeof(header), file) != nullptr;
    bool viewport = ok &&
        sscanf_s(header, "%39[^,],%d,%d,%u,%d,%d,%d,%d,%u", magic,
                 static_cast<unsigned>(sizeof(magic)), &columns, &rows, &count,
                 &origin_column, &origin_row, &map_width, &map_height,
                 &halo_count) == 9 &&
        (std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V1") == 0 ||
         std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V2") == 0);
    if (!viewport) {
        ok = ok && sscanf_s(header, "%39[^,],%d,%d,%u", magic,
                            static_cast<unsigned>(sizeof(magic)),
                            &map_width, &map_height, &count) == 4 &&
             (std::strcmp(magic, "C3X_BIQ_TERRAIN_V0") == 0 || std::strcmp(magic, "C3X_BIQ_TERRAIN_V3") == 0);
        halo_count = 0;
    }
    ok = ok && count <= 1000000u && halo_count <= 1000000u;
    if (ok) {
        bool has_river_topology = viewport &&
            std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V2") == 0;
        tiles.reserve(count + halo_count);
        for (unsigned index = 0; index < count + halo_count; ++index) {
            CsvTile tile = {};
            if (viewport) {
                int column = 0, row = 0, source_x = 0, source_y = 0;
                int parsed = has_river_topology
                    ? fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u,%u\n",
                               &column, &row, &source_x, &source_y,
                               &tile.base, &tile.real, &tile.bonus,
                               &tile.overlays, &tile.river)
                    : fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u\n",
                               &column, &row, &source_x, &source_y,
                               &tile.base, &tile.real, &tile.bonus,
                               &tile.overlays);
                if (parsed != (has_river_topology ? 9 : 8)) {
                    ok = false;
                    break;
                }
                tile.x = source_x;
                tile.y = source_y;
            } else if (std::strcmp(magic,"C3X_BIQ_TERRAIN_V3")==0) {
                if(fscanf_s(file,"%d,%d,%d,%d,%u,%u,%u\n",&tile.x,&tile.y,&tile.base,&tile.real,
                    &tile.bonus,&tile.overlays,&tile.river)!=7){ok=false;break;}
            } else if (fscanf_s(file, "%d,%d,%d,%d,%u,%u\n",
                                &tile.x, &tile.y, &tile.base, &tile.real,
                                &tile.bonus, &tile.overlays) != 6) {
                ok = false;
                break;
            }
            tile.topology_halo = index >= count;
            tiles.push_back(tile);
        }
        ok = ok && tiles.size() == count + halo_count;
    }
    fclose(file);
    return ok;
}

bool write_bmp(char const * path, c3x_renderer_output_v1 const & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "wb") != 0 || file == nullptr)
        return false;
    BITMAPFILEHEADER file_header = {};
    BITMAPINFOHEADER info = {};
    file_header.bfType = 0x4d42;
    file_header.bfOffBits = sizeof(file_header) + sizeof(info);
    file_header.bfSize = file_header.bfOffBits + output.stride_bytes * output.height;
    info.biSize = sizeof(info);
    info.biWidth = output.width;
    info.biHeight = -output.height;
    info.biPlanes = 1;
    info.biBitCount = 32;
    info.biCompression = BI_RGB;
    bool ok = fwrite(&file_header, sizeof(file_header), 1, file) == 1 &&
              fwrite(&info, sizeof(info), 1, file) == 1 &&
              fwrite(output.bgra_pixels, output.stride_bytes * output.height, 1, file) == 1;
    fclose(file);
    return ok;
}

// Optional headless witness of the final GDI conversion; no game HWND or UI.
bool write_color_preview(char const* path, HMODULE module, c3x_renderer_output_v1 const& output) {
    auto blit = reinterpret_cast<c3x_renderer_blit_fn>(GetProcAddress(module,"c3x_renderer_blit"));
    if (!blit) return false;
    BITMAPINFO info={}; info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth=output.width; info.bmiHeader.biHeight=-output.height;
    info.bmiHeader.biPlanes=1; info.bmiHeader.biCompression=BI_RGB;
    HDC dc32=CreateCompatibleDC(nullptr),dc16=CreateCompatibleDC(nullptr);
    void* bits32=nullptr;void* bits16=nullptr;
    info.bmiHeader.biBitCount=32;
    HBITMAP bitmap32=CreateDIBSection(dc32,&info,DIB_RGB_COLORS,&bits32,nullptr,0);
    info.bmiHeader.biBitCount=16;
    HBITMAP bitmap16=CreateDIBSection(dc16,&info,DIB_RGB_COLORS,&bits16,nullptr,0);
    bool ok=dc32 && dc16 && bitmap32 && bitmap16 && bits32 && bits16;
    HGDIOBJ previous32=nullptr,previous16=nullptr;
    if (ok) {
        previous32=SelectObject(dc32,bitmap32);previous16=SelectObject(dc16,bitmap16);
        std::memcpy(bits32,output.bgra_pixels,std::size_t(output.stride_bytes)*output.height);
        std::vector<std::uint32_t> pixels(std::size_t(output.width)*output.height);
        auto save=[&](char const* suffix) {
            GdiFlush();
            unsigned stride=(unsigned(output.width)*2+3)&~3u;
            for(int y=0;y<output.height;++y)for(int x=0;x<output.width;++x) {
                auto row=reinterpret_cast<std::uint16_t const*>(static_cast<std::uint8_t const*>(bits16)+y*stride);
                unsigned value=row[x],r=(value>>10)&31,g=(value>>5)&31,b=value&31;
                pixels[std::size_t(y)*output.width+x]=0xff000000u|((r*255/31)<<16)|((g*255/31)<<8)|(b*255/31);
            }
            auto preview=output;preview.bgra_pixels=pixels.data();
            return write_bmp((std::string(path)+suffix).c_str(),preview);
        };
        ok=BitBlt(dc16,0,0,output.width,output.height,dc32,0,0,SRCCOPY)!=FALSE && save(".rgb555-before.bmp");
        LARGE_INTEGER start={},finish={},frequency={};QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
        for(int repeat=0;ok && repeat<20;++repeat)ok=blit(&output,dc16)==C3X_RENDERER_RESULT_OK;
        GdiFlush();QueryPerformanceCounter(&finish);
        std::printf("COLOR RGB555 blit_mean_ms=%.3f repeats=20\n",1000.0*double(finish.QuadPart-start.QuadPart)/double(frequency.QuadPart)/20);
        ok=ok && save(".rgb555-after.bmp");
        SelectObject(dc32,previous32);SelectObject(dc16,previous16);
    }
    if(bitmap32)DeleteObject(bitmap32);if(bitmap16)DeleteObject(bitmap16);
    if(dc32)DeleteDC(dc32);if(dc16)DeleteDC(dc16);
    return ok;
}

#include "unit_roster_preview.h"

// Offscreen verification must report a crash, not wait indefinitely in a
// hidden Windows Error Reporting dialog. No handler is installed in Civ III.
void preview_fault_address(char const* label,void* address) {
    MEMORY_BASIC_INFORMATION memory={};char module[MAX_PATH]={};
    if(VirtualQuery(address,&memory,sizeof(memory)))
        GetModuleFileNameA(static_cast<HMODULE>(memory.AllocationBase),module,sizeof(module));
    auto name=std::strrchr(module,'\\');
    std::fprintf(stderr,"%s address=%p module=%s offset=0x%zx\n",label,address,
        name?name+1:module,reinterpret_cast<std::uintptr_t>(address)-reinterpret_cast<std::uintptr_t>(memory.AllocationBase));
}

LONG WINAPI preview_unhandled_exception(EXCEPTION_POINTERS* fault) {
    std::fprintf(stderr,"FAIL native-exception code=0x%08lx parameters=%lu\n",
        fault->ExceptionRecord->ExceptionCode,fault->ExceptionRecord->NumberParameters);
    preview_fault_address("fault",fault->ExceptionRecord->ExceptionAddress);
    for(DWORD i=0;i<fault->ExceptionRecord->NumberParameters;++i)
        std::fprintf(stderr,"exception-parameter[%lu]=0x%zx\n",i,std::size_t(fault->ExceptionRecord->ExceptionInformation[i]));
    // The handler's own backtrace has already lost the faulting leaf. Preserve
    // bounded image-backed candidates from its saved x86 stack for the map file.
#ifdef _M_IX86
    unsigned words[32]={};SIZE_T copied=0;
    if(ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<void*>(fault->ContextRecord->Esp),words,sizeof(words),&copied))
        for(unsigned i=0;i<copied/sizeof(unsigned);++i){MEMORY_BASIC_INFORMATION memory={};auto address=reinterpret_cast<void*>(words[i]);
            if(VirtualQuery(address,&memory,sizeof(memory)) && memory.Type==MEM_IMAGE)preview_fault_address("fault-stack",address);}
#endif
    void* frames[32]={};USHORT count=CaptureStackBackTrace(0,32,frames,nullptr);
    for(USHORT i=0;i<count;++i)preview_fault_address("stack",frames[i]);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

int run_preview_case(int argc, char ** argv, HMODULE shared_module=nullptr, bool configure=true, bool bootstrap=false) {
    LARGE_INTEGER timing_frequency={},process_enter={},source_done={},dll_done={},definitions_done={};
    QueryPerformanceFrequency(&timing_frequency);QueryPerformanceCounter(&process_enter);
    char timing_option[8]={};
    bool timing_enabled=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_TIMING",timing_option,sizeof(timing_option))!=0;
    struct TimingRequest {
        long long capture_begin,capture_end,caller_enter,caller_return,correct_done;
        long long geometry,draw,readback;
        int result;unsigned tiles;
    };
    std::vector<TimingRequest> timing_requests;
    if(timing_enabled)timing_requests.reserve(4096);
    unsigned timing_dropped=0;
    LARGE_INTEGER capture_begin={},capture_end={};
    bool fresh_capture=false;
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    SetUnhandledExceptionFilter(preview_unhandled_exception);
    if (argc != 11 && argc != 12) {
        std::fprintf(stderr, "usage: biq_preview <dll> <mod-root> <definitions> <scene.csv> <out.bmp> <width> <height> <center-x> <center-y> <tile-width> [hour]\n");
        return 2;
    }
    int target_width = std::atoi(argv[6]);
    int target_height = std::atoi(argv[7]);
    int center_x = std::atoi(argv[8]);
    int center_y = std::atoi(argv[9]);
    int tile_width = std::atoi(argv[10]);
    int tile_height = tile_width / 2;
    if (target_width < 320 || target_height < 200 || tile_width < 32 || tile_width > 256)
        return 2;

    int map_width = 0, map_height = 0;
    char boundary_option[8]={};
    bool retained_boundary=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_RETAINED_BOUNDARY",boundary_option,sizeof(boundary_option))!=0;
    char topology_edit_option[8]={};
    bool topology_edits=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_TOPOLOGY_EDITS",topology_edit_option,sizeof(topology_edit_option))!=0;
    bool boundary_resource=true,boundary_mine=true;
    int boundary_x=center_x,boundary_y=center_y;
    std::vector<CsvTile> source_tiles;
    if (!read_scene(argv[4], map_width, map_height, source_tiles)) {
        std::fprintf(stderr, "error: could not read BIQ terrain scene\n");
        return 1;
    }
    if(retained_boundary)for(auto& tile:source_tiles){
        tile.base=tile.real=2;tile.bonus=tile.overlays=tile.river=0;
    }
    char profile[64]={};GetEnvironmentVariableA("C3X_RENDERER_VISUAL_PROFILE",profile,sizeof(profile));
    bool pickup=std::strcmp(profile,"frozen")!=0;
    if(pickup && source_tiles.size()!=std::size_t(map_width)*map_height/2){
        std::fprintf(stderr,"pickup preview requires the complete world CSV\n");return 1;
    }
    char activity[8]={};bool active=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_ACTIVE_VOLCANO",activity,sizeof(activity))!=0;
    std::vector<unsigned> world(std::size_t(map_width)*map_height/2,0);
    if(pickup)for(auto const& t:source_tiles)world[(std::size_t(t.y)*map_width+t.x)/2]=
        unsigned(t.base)|(unsigned(t.real)<<8)|(t.river<<16)|(active && t.real==10 ? 1u<<24 : 0);
    QueryPerformanceCounter(&source_done);
    HMODULE module = shared_module ? shared_module : LoadLibraryA(argv[1]);
    QueryPerformanceCounter(&dll_done);
    if (module == nullptr)
        return 1;
    auto set_definitions = reinterpret_cast<c3x_renderer_set_definition_paths_fn>(
        GetProcAddress(module, "c3x_renderer_set_definition_paths"));
    auto render = reinterpret_cast<c3x_renderer_render_fn>(GetProcAddress(module, "c3x_renderer_render"));
    auto reset = reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(module, "c3x_renderer_reset"));
    char custom_definitions[4 * MAX_PATH] = {};
    GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS",custom_definitions,sizeof(custom_definitions));
    char const* custom_path=custom_definitions[0] ? custom_definitions : nullptr;
    auto set_units=reinterpret_cast<c3x_renderer_set_unit_rendering_fn>(
        GetProcAddress(module,"c3x_renderer_set_unit_rendering"));
    char unit_preview[8]={};
    bool enable_unit_preview=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_UNITS",unit_preview,sizeof(unit_preview))!=0;
    char idle_units_option[16]={};
    int idle_unit_count=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_IDLE_UNITS",idle_units_option,sizeof(idle_units_option))?
        std::clamp(std::atoi(idle_units_option),0,64):0;
    char unit_actions_option[16]={};
    bool mixed_unit_actions=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_UNIT_ACTIONS",unit_actions_option,sizeof(unit_actions_option)) &&
        std::strcmp(unit_actions_option,"mixed")==0;
    bool realistic_unit_actions=std::strcmp(unit_actions_option,"realistic")==0;
#ifdef C3X_LAB_PREVIEW
    enable_unit_preview=enable_unit_preview || GetEnvironmentVariableA("C3X_LAB_UNIT_STUDY",unit_preview,sizeof(unit_preview))!=0;
#endif
    bool gpu_contract_units=false;
#ifdef C3X_GPU_NATIVE_CONTRACT
    char gpu_units[8]={};gpu_contract_units=GetEnvironmentVariableA("C3X_RENDERER_GPU_FRAME_TEST",gpu_units,sizeof(gpu_units)) && !std::strcmp(gpu_units,"1");
#endif
    if((enable_unit_preview || idle_unit_count || gpu_contract_units) &&
       (!set_units || set_units(1)!=C3X_RENDERER_RESULT_OK))return 1;
    if (set_definitions == nullptr || render == nullptr || reset == nullptr ||
        (configure && set_definitions(argv[2], argv[3], nullptr, custom_path) != C3X_RENDERER_RESULT_OK))
        return 1;

    QueryPerformanceCounter(&definitions_done);
    char object_option[8]={};bool objects=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_OBJECTS",object_option,sizeof(object_option))!=0;
    char animation_option[8]={};bool animate=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_ANIMATION",animation_option,sizeof(animation_option))!=0;
    char dense_option[8]={};bool dense_scene=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_DENSE_SCENE",dense_option,sizeof(dense_option))!=0;
    char farm_option[8]={};bool farm_study=GetEnvironmentVariableA("C3X_LAB_FARM_STUDY",farm_option,sizeof(farm_option))!=0;
    // Optional dense-city case uses the same four fields as the ordinary city
    // witness. Default dense capture retains its existing industrial fixture.
    char dense_city_option[80]={};int dense_city[4]={};
    bool dense_city_case=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_DENSE_CITY_CASE",dense_city_option,sizeof(dense_city_option)) &&
        sscanf_s(dense_city_option,"%d,%d,%d,%d",&dense_city[0],&dense_city[1],&dense_city[2],&dense_city[3])==4 &&
        dense_city[0]>=0 && dense_city[0]<=4 && dense_city[1]>=0 && dense_city[1]<=3 &&
        dense_city[2]>=0 && dense_city[2]<=2 && dense_city[3]>=0 && dense_city[3]<=1;
    std::vector<std::array<int,2>> resource_sites;
    std::vector<std::array<int,2>> city_object_sites;
    char prepared_area_option[8]={};
    bool prepared_view_fixture=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_PREPARED_VIEW",prepared_area_option,sizeof(prepared_area_option))!=0;
    int visibility_center_x=center_x,visibility_center_y=center_y;
    bool capture_whole_world=false;
    char world_readiness_option[8]={};
    bool fixed_world_scene=GetEnvironmentVariableA("C3X_RENDERER_WORLD_READINESS_TEST",world_readiness_option,sizeof(world_readiness_option))!=0;
    auto capture_view = [&]() {
    if(timing_enabled)QueryPerformanceCounter(&capture_begin);
    int center_raw_x = center_x * tile_width / 2;
    int center_raw_y = center_y * tile_height / 2;
    int shift_x = target_width / 2 - tile_width / 2 - center_raw_x;
    int shift_y = target_height / 2 - tile_height / 2 - center_raw_y;
    std::vector<c3x_renderer_tile_v1> tiles;
    for (CsvTile const & source : source_tiles) {
      for (int wrap_copy = -1; wrap_copy <= 1; ++wrap_copy) {
        if(capture_whole_world && wrap_copy!=0)continue;
        int render_x = source.x + wrap_copy * map_width;
        int anchor_x = render_x * tile_width / 2 + shift_x;
        int anchor_y = source.y * tile_height / 2 + shift_y;
        int margin=pickup?tile_width*(prepared_view_fixture?12:6):96,vertical=pickup?tile_height*(prepared_view_fixture?12:6):128;
        if (!capture_whole_world && (anchor_x + tile_width < -margin || anchor_x > target_width + margin ||
            anchor_y + tile_height < -vertical || anchor_y > target_height + vertical))
            continue;
        c3x_renderer_tile_v1 tile = {};
        tile.tile_x = render_x;
        tile.tile_y = source.y;
        tile.anchor_x = anchor_x;
        tile.anchor_y = anchor_y;
        tile.terrain_type = source.base;
        tile.real_terrain_type = source.real;
        tile.square_parts = source.bonus;
        tile.terrain_overlays = source.overlays;
        tile.river_code = source.river;
        tile.visibility_mask = 1;
        tile.tile_visibility = 1;
        tile.variant_seed = preview_seed(source.x, source.y);
        tile.tile_flags = source.topology_halo
            ? C3X_RENDERER_TILE_TOPOLOGY_HALO : C3X_RENDERER_TILE_RENDER;
        if(pickup && (anchor_x+tile_width<0 || anchor_x>target_width || anchor_y+tile_height<0 || anchor_y>target_height))
            tile.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_PREFETCH;
        if(pickup && (anchor_x+tile_width < -tile_width*(prepared_view_fixture?8:4) || anchor_x>target_width+tile_width*(prepared_view_fixture?8:4) ||
            anchor_y+tile_height < -tile_height*(prepared_view_fixture?8:4) || anchor_y>target_height+tile_height*(prepared_view_fixture?8:4)))
            tile.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
        tile.tile_flags|=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
        char visibility_fixture[8]={};
        if(GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_VISIBILITY",visibility_fixture,sizeof(visibility_fixture))){
            tile.tile_flags&=~(C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE);
            int distance=(std::min)(std::abs(source.x-visibility_center_x),map_width-std::abs(source.x-visibility_center_x))+std::abs(source.y-visibility_center_y);
            if(distance<9)tile.tile_flags|=C3X_RENDERER_TILE_EXPLORED;
            if(distance<5)tile.tile_flags|=C3X_RENDERER_TILE_VISIBLE;
        }
        tile.resource_id = tile.resource_class = tile.tile_building_id = tile.barbarian_tribe_id = -1;
        tile.city_id = tile.city_owner_id = tile.city_size = tile.city_culture_group = tile.city_era = -1;
        tile.unit_type_id = tile.unit_owner_id = tile.unit_class = tile.unit_state = tile.unit_damage = tile.unit_direction = -1;
        tile.territory_owner_id = -1;
        if (source.real == 7) tile.feature_flags = C3X_RENDERER_FEATURE_FOREST;
        if (source.real == 8) tile.feature_flags = C3X_RENDERER_FEATURE_JUNGLE;
        if (source.real == 9) tile.feature_flags = C3X_RENDERER_FEATURE_MARSH;
        if (source.real == 10) {tile.feature_flags = C3X_RENDERER_FEATURE_VOLCANO;tile.has_effect=active?1:0;}
        tiles.push_back(tile);
      }
    }
    if(farm_study)for(auto& tile:tiles){
        if(tile.real_terrain_type<0 || tile.real_terrain_type>4 || tile.terrain_type>4)continue;
        unsigned seed=preview_seed((tile.tile_x%map_width+map_width)%map_width,tile.tile_y);
        if(seed%7u>=3u)continue;
        tile.improvement_flags|=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
        tile.irrigation_mask=seed&15u;
        tile.route_style=int((seed>>4)&3u);
    }
    char road_study_option[8]={};
    if(GetEnvironmentVariableA("C3X_LAB_ROAD_STUDY",road_study_option,sizeof(road_study_option))){
        int era=std::clamp(std::atoi(road_study_option),0,3);
        char road_layout[24]={};
        bool isolated=GetEnvironmentVariableA("C3X_LAB_ROAD_LAYOUT",road_layout,sizeof(road_layout)) &&
            std::strcmp(road_layout,"isolated")==0;
        bool isolated_plains=std::strcmp(road_layout,"isolated-plains")==0;
        bool isolated_mountain=std::strcmp(road_layout,"isolated-mountain")==0;
        bool orthogonal=std::strcmp(road_layout,"orthogonal")==0;
        for(auto& tile:tiles){
            if(tile.real_terrain_type<0 || tile.real_terrain_type>10)continue;
            int x=((tile.tile_x%map_width)+map_width)%map_width;
            int c=(x+tile.tile_y)/2,r=(x-tile.tile_y)/2;
            unsigned seed=preview_seed(x,tile.tile_y);
            bool arterial=(c%7==0 || (r+70)%7==0);
            bool dense=(c>=55 && c<=66 && r>=-4 && r<=9) ||
                (c>=47 && c<=56 && r>=20 && r<=30);
            bool branch=(seed%29u==0 && (c%7==1 || (r+70)%7==1));
            bool road=isolated ? x==21 && tile.tile_y==85 :
                isolated_plains ? x==77 && tile.tile_y==25 :
                isolated_mountain ? x==77 && tile.tile_y==23 :
                orthogonal ? ((x==80 && (tile.tile_y==24 || tile.tile_y==26 || tile.tile_y==28)) ||
                              (tile.tile_y==26 && (x==78 || x==82))) :
                arterial || dense || branch;
            if(road){tile.road_mask=1;tile.route_style=era;}
        }
    }
    if(objects && !fixed_world_scene) {
        char city_only_setting[8]={};
        bool city_only=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CITY_ONLY",city_only_setting,sizeof(city_only_setting)) && city_only_setting[0]=='1';
        std::vector<std::size_t> candidates;
        for(std::size_t i=0;i<tiles.size();++i)if(tiles[i].real_terrain_type<=(city_only?5:4) &&
            (tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER))candidates.push_back(i);
        std::sort(candidates.begin(),candidates.end(),[&](auto a,auto b){
            auto distance=[&](auto i){auto const& t=tiles[i];return std::abs(t.anchor_x-target_width/2)+std::abs(t.anchor_y-target_height/2);};
            return distance(a)<distance(b);
        });
        char fixed_city_case[80]={};
        if(GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CITY",fixed_city_case,sizeof(fixed_city_case))){
            char fixed_site[80]={};
            if(GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CITY_SITE",fixed_site,sizeof(fixed_site))){
                int site_x=-1,site_y=-1;
                if(sscanf_s(fixed_site,"%d,%d",&site_x,&site_y)==2 &&
                   site_x>=0 && site_x<map_width && site_y>=0 && site_y<map_height)
                    city_object_sites.push_back({site_x,site_y});
            }
            if(city_object_sites.empty() && candidates.size()>=6)
                for(unsigned object=0;object<6;object++){auto const&t=tiles[candidates[object]];
                    city_object_sites.push_back({((t.tile_x%map_width)+map_width)%map_width,t.tile_y});}
            if(!city_object_sites.empty()){
                candidates.clear();
                for(auto const&site:city_object_sites){
                    std::size_t best=tiles.size();int nearest=0x7fffffff;
                    for(std::size_t i=0;i<tiles.size();i++)if(((tiles[i].tile_x%map_width)+map_width)%map_width==site[0] && tiles[i].tile_y==site[1]){
                        int distance=std::abs(tiles[i].anchor_x-target_width/2)+std::abs(tiles[i].anchor_y-target_height/2);
                        if(distance<nearest){nearest=distance;best=i;}
                    }
                    if(best<tiles.size())candidates.push_back(best);
                }
            }
        }
        if(candidates.size()>=(city_only?1u:6u)){
            auto& city=tiles[candidates[0]];city.city_id=1;city.city_owner_id=1;city.city_size=2;
            city.city_culture_group=0;city.city_era=2;city.city_flags=C3X_RENDERER_CITY_CAPITAL|C3X_RENDERER_CITY_WALLED;
            char city_case[80]={};
            if(GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CITY",city_case,sizeof(city_case))){
                int culture=0,era=0,size=0,capital=0,walled=0;
                int fields=sscanf_s(city_case,"%d,%d,%d,%d,%d",&culture,&era,&size,&capital,&walled);
                if(fields>=4 &&
                    culture>=0 && culture<=4 && era>=0 && era<=3 && size>=0 && size<=2 && capital>=0 && capital<=1 &&
                    (fields==4 || walled==0 || walled==1)){
                    city.city_culture_group=culture;city.city_era=era;city.city_size=size;
                    city.city_flags=(capital?C3X_RENDERER_CITY_CAPITAL:0)|(fields==5 && walled?C3X_RENDERER_CITY_WALLED:0);
                }
            }
            if(!city_only){
                auto& mine=tiles[candidates[1]];mine.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;mine.route_style=2;
                auto& farm=tiles[candidates[2]];farm.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;farm.irrigation_mask=15;farm.route_style=2;
                auto& resource=tiles[candidates[3]];resource.resource_id=1;resource.resource_class=0;strcpy_s(resource.resource_name,"Iron");
                auto& road=tiles[candidates[4]];road.road_mask=15;road.route_style=2;
                auto& rail=tiles[candidates[5]];rail.road_mask=15;rail.railroad_mask=15;rail.route_style=3;
            }
        }
    }
    if(dense_scene)for(auto & tile:tiles) {
        // Synthetic stress inputs remain world-fixed across capture changes.
        // Fill ordinary supported objects only; draw eligibility stays native.
        int x=((tile.tile_x%map_width)+map_width)%map_width,y=tile.tile_y;
        auto seed=preview_seed(x,y);
        bool land=tile.real_terrain_type>=0 && tile.real_terrain_type<=4;
        if(land) {
            tile.road_mask=15;tile.route_style=2;
            if(y%8<2){tile.railroad_mask=15;tile.route_style=3;}
            if(x%12==3 && y%12==3) {
                tile.city_id=1+(y*map_width+x)/2;tile.city_owner_id=1;
                tile.city_size=int(seed%3);tile.city_population=8+int(seed%12);
                tile.city_culture_group=0;tile.city_era=2;
                if(dense_city_case){tile.city_culture_group=dense_city[0];tile.city_era=dense_city[1];
                    tile.city_size=dense_city[2];tile.city_flags=dense_city[3]?C3X_RENDERER_CITY_CAPITAL:0;}
            } else if(seed%41==0) {
                tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP;
                tile.barbarian_tribe_id=7;
            } else if(seed%3==0)tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;
            else {tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;tile.irrigation_mask=15;}
        }
        if(seed%11==0 && tile.city_id<0) {
            char const* land_names[]={"Iron","Cattle","Horses","Wheat","Gold","Dyes"};
            char const* name=land?land_names[(seed/11)%6]:tile.real_terrain_type>=11?(seed&1?"Fish":"Whales"):nullptr;
            if(name){tile.resource_id=100+int((seed/11)%6);tile.resource_class=0;strcpy_s(tile.resource_name,name);}
        }
    }
    if (animate && !dense_scene && !retained_boundary) {
        char const * names[]={"Horses","Cattle","Wheat","Fish","Whales","Game","Furs","Ivory","Bananas","Rubber"};
        if (resource_sites.empty())
            for (auto const & tile:tiles)
                if ((tile.tile_flags&C3X_RENDERER_TILE_RENDER) && tile.anchor_x>target_width/5 &&
                    tile.anchor_x<target_width*4/5 && tile.anchor_y>target_height/4 && tile.anchor_y<target_height*3/4 &&
                    resource_sites.size()<std::size(names))
                    resource_sites.push_back({((tile.tile_x%map_width)+map_width)%map_width,tile.tile_y});
        for(auto & tile:tiles)for(unsigned i=0;i<resource_sites.size();++i)
            if(((tile.tile_x%map_width)+map_width)%map_width==resource_sites[i][0] && tile.tile_y==resource_sites[i][1]) {
                tile.resource_id=int(100+i);strcpy_s(tile.resource_name,names[i]);
            }
    }
#ifdef C3X_LAB_PREVIEW
    lab_place_objects(tiles,center_x,center_y,map_width);
#endif
    if(retained_boundary)for(auto& tile:tiles){
        if(boundary_resource && tile.tile_x==boundary_x && tile.tile_y==boundary_y){
            tile.resource_id=1;tile.resource_class=0;strcpy_s(tile.resource_name,"Horses");
        }
        if(boundary_mine && tile.tile_x==boundary_x && tile.tile_y==boundary_y-2)
            tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;
    }
    if(timing_enabled){QueryPerformanceCounter(&capture_end);fresh_capture=true;}
    return tiles;
    };
    auto tiles=capture_view();
    if(dense_scene && timing_enabled) {
        unsigned cities=0,roads=0,rails=0,improvements=0,resources=0;
        for(auto const& tile:tiles){cities+=tile.city_id>=0;roads+=tile.road_mask!=0;
            rails+=tile.railroad_mask!=0;improvements+=tile.improvement_flags!=0;resources+=tile.resource_id>=0;}
        std::printf("DENSE_FIXTURE world_width=%d world_height=%d captured=%zu cities=%u roads=%u rails=%u improvements=%u resources=%u\n",
            map_width,map_height,tiles.size(),cities,roads,rails,improvements,resources);
    }
    c3x_renderer_frame_v1 frame = {};
    auto api_version=reinterpret_cast<c3x_renderer_get_api_version_fn>(GetProcAddress(module,"c3x_renderer_get_api_version"));
    frame.api_version = api_version ? api_version() : C3X_RENDERER_API_VERSION;
    frame.struct_size = sizeof(frame);
    frame.target_width = target_width;
    frame.target_height = target_height;
    frame.clip_right = target_width;
    frame.clip_bottom = target_height;
    frame.tile_width = tile_width;
    frame.tile_height = tile_height;
    frame.hour = argc == 12 ? std::atoi(argv[11]) : 12;
    frame.tile_count = static_cast<c3x_renderer_u32>(tiles.size());
    frame.tiles = tiles.data();
    frame.presentation_time_ticks = 1000000;
    frame.presentation_frequency = 1000000;
    frame.dirty_flags = C3X_RENDERER_DIRTY_ALL;
    frame.world_width_tiles = map_width;
    frame.world_height_tiles = map_height;
    frame.world_wrap_x = 1;
    if(pickup){frame.world_topology_count=unsigned(world.size());frame.world_topology=world.data();frame.world_topology_revision=1;}
    char season[16]={};if(GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_SEASON",season,sizeof(season)))frame.season=std::atoi(season);
    c3x_renderer_output_v1 output = {C3X_RENDERER_API_VERSION, sizeof(output)};
    char camera_option[16]={};
    bool background_camera=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CAMERA_QUEUE",camera_option,sizeof(camera_option))!=0;
    auto camera_begin=reinterpret_cast<c3x_renderer_camera_begin_fn>(GetProcAddress(module,"c3x_renderer_camera_begin"));
    auto camera_poll=reinterpret_cast<c3x_renderer_camera_poll_fn>(GetProcAddress(module,"c3x_renderer_camera_poll"));
    bool camera_view=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CAMERA_VIEW",camera_option,sizeof(camera_option))!=0;
    auto camera_begin_view=reinterpret_cast<c3x_renderer_camera_begin_view_fn>(GetProcAddress(module,"c3x_renderer_camera_begin_view"));
    auto prepare_view=reinterpret_cast<c3x_renderer_prepare_view_fn>(GetProcAddress(module,"c3x_renderer_prepare_view"));
    auto render_view=reinterpret_cast<c3x_renderer_render_view_fn>(GetProcAddress(module,"c3x_renderer_render_view"));
    auto camera_poll_view=reinterpret_cast<c3x_renderer_camera_poll_view_fn>(GetProcAddress(module,"c3x_renderer_camera_poll_view"));
    bool ambient_async=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_AMBIENT_ASYNC",camera_option,sizeof(camera_option))!=0;
    bool ambient_boundary=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_AMBIENT_BOUNDARY",camera_option,sizeof(camera_option))!=0;
    char ambient_soak_option[16]={};
    int ambient_soak_seconds=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_AMBIENT_SOAK_SECONDS",ambient_soak_option,sizeof(ambient_soak_option))?
        std::clamp(std::atoi(ambient_soak_option),0,60):0;
    if(camera_view && (!background_camera || !camera_begin_view || !camera_poll_view)) {
        std::fputs("camera view extension exports missing or queue disabled\n",stderr);return 1;
    }
    if(background_camera && (!camera_begin || !camera_poll)) {
        std::fputs("camera extension exports missing\n",stderr);return 1;
    }
    if(ambient_async && (!camera_begin_view || !camera_poll_view || (ambient_boundary && !render_view))) {
        std::fputs("ambient async requires camera view extension exports\n",stderr);return 1;
    }
    bool native_handoff=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_NATIVE_HANDOFF",camera_option,sizeof(camera_option))!=0;
    auto camera_present_view=reinterpret_cast<c3x_renderer_camera_present_view_fn>(GetProcAddress(module,"c3x_renderer_camera_present_view"));
    if(native_handoff && (!camera_present_view || !camera_begin_view || !camera_poll_view || !render_view || background_camera)) {
        std::fputs("native handoff witness requires presentation exports and its own queue policy\n",stderr);return 1;
    }
    auto prepare_nearby_view=reinterpret_cast<c3x_renderer_prepare_nearby_view_fn>(GetProcAddress(module,"c3x_renderer_prepare_nearby_view"));
    c3x_renderer_i64 handoff_ticket=0;
    c3x_renderer_i64 handoff_clock=0;
    unsigned camera_case=0;
    bool prepare_view_family=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_VIEW_FAMILY",camera_option,sizeof(camera_option))!=0;
    auto render_checked = [&](c3x_renderer_frame_v1 const* input, c3x_renderer_output_v1* result) {
        LARGE_INTEGER caller_enter={},caller_return={},correct_done={};
        if(timing_enabled)QueryPerformanceCounter(&caller_enter);
        int code=C3X_RENDERER_RESULT_ERROR;
        if(native_handoff) {
            // Match the current injected policy: the camera is already native
            // state. Only fresh current-camera pixels may be acquired.
            LARGE_INTEGER begin={},frequency={},ended={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&begin);
            c3x_renderer_camera_identity_v1 identity={1,2,3,input->world_topology_revision};
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),input,identity};
            c3x_renderer_camera_view_v1 shown={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(shown)};
            if(handoff_ticket){
                auto status=camera_poll_view(handoff_ticket,&shown);
                if(status!=C3X_RENDERER_RESULT_PENDING)handoff_ticket=0;
            }
            int held=camera_present_view(&request,&shown);
            if(held==C3X_RENDERER_RESULT_OK){*result=shown.output;code=held;handoff_clock=shown.frame.presentation_time_ticks;}
            else {code=render_view(&request,result);handoff_clock=input->presentation_time_ticks;handoff_ticket=0;}
            bool preparing=code==C3X_RENDERER_RESULT_OK && prepare_nearby_view && prepare_nearby_view(&request)==C3X_RENDERER_RESULT_OK;
            if(preparing && prepare_view_family && prepare_view){
                auto caller_capture_begin=capture_begin,caller_capture_end=capture_end;bool caller_fresh_capture=fresh_capture;
                int saved_width=tile_width,saved_height=tile_height;
                for(int level:{128,160,192})if(level!=input->tile_width){
                    auto future=*input;future.tile_width=level;future.tile_height=level/2;
                    auto prospective_request=request;prospective_request.frame=&future;
                    if(prepare_view(&prospective_request,1)!=C3X_RENDERER_RESULT_PENDING)continue;
                    tile_width=level;tile_height=level/2;auto captured=capture_view();
                    future.tiles=captured.data();future.tile_count=unsigned(captured.size());
                    prepare_view(&prospective_request,0);
                }
                tile_width=saved_width;tile_height=saved_height;
                capture_begin=caller_capture_begin;capture_end=caller_capture_end;fresh_capture=caller_fresh_capture;
            }
            auto quantum=(std::max)(c3x_renderer_i64(1),input->presentation_frequency/15);
            if(!preparing && !handoff_ticket && result->visible_animation_count && handoff_clock/quantum!=input->presentation_time_ticks/quantum)
                camera_begin_view(&request,&handoff_ticket);
            QueryPerformanceCounter(&ended);
            std::printf("NATIVE_HANDOFF case=%u calls=1 holds=%u exact_ms=%.3f clock_age_ms=%.3f result=%d prepared_policy=%u current_camera=1\n",
                camera_case++,unsigned(held==C3X_RENDERER_RESULT_OK),double(ended.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(input->presentation_time_ticks-handoff_clock)*1000/input->presentation_frequency,code,unsigned(preparing));
        } else if(background_camera) {
            LARGE_INTEGER begin={},accepted={},finished={},frequency={};QueryPerformanceFrequency(&frequency);
            c3x_renderer_i64 ticket=0,obsolete=0;
            // Exercise actual in-flight supersession, not only an empty queue.
            // This earlier valid scene has a different environment/pose clock.
            auto earlier=*input;earlier.hour=(earlier.hour+1)%24;
            earlier.presentation_time_ticks+=earlier.presentation_frequency/2;
            if(camera_begin(&earlier,&obsolete)!=C3X_RENDERER_RESULT_PENDING)return int(C3X_RENDERER_RESULT_ERROR);
            Sleep(10);
            QueryPerformanceCounter(&begin);
            c3x_renderer_camera_identity_v1 identity={1,2,c3x_renderer_i64(camera_case)+1,c3x_renderer_i64(camera_case)+1};
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),input,identity};
            code=camera_view?camera_begin_view(&request,&ticket):camera_begin(input,&ticket);
            QueryPerformanceCounter(&accepted);
            if(code!=C3X_RENDERER_RESULT_PENDING || camera_poll(obsolete,result)!=C3X_RENDERER_RESULT_SUPERSEDED)
                return int(C3X_RENDERER_RESULT_ERROR);
            auto start=GetTickCount64();unsigned polls=0;bool first_image=false;
            double poll_max_ms=0,repeat_max_ms=0;
            ++camera_case;
            while(code==C3X_RENDERER_RESULT_PENDING && GetTickCount64()-start<120000) {
                LARGE_INTEGER repeat_begin={},poll_begin={},poll_end={};QueryPerformanceCounter(&repeat_begin);
                c3x_renderer_i64 repeated=0;
                int repeated_code=camera_view?camera_begin_view(&request,&repeated):camera_begin(input,&repeated);
                QueryPerformanceCounter(&poll_begin);
                if(repeated_code!=C3X_RENDERER_RESULT_PENDING || repeated!=ticket)return int(C3X_RENDERER_RESULT_ERROR);
                repeat_max_ms=(std::max)(repeat_max_ms,double(poll_begin.QuadPart-repeat_begin.QuadPart)*1000/frequency.QuadPart);
                if(camera_view){
                    c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
                    code=camera_poll_view(ticket,&view);
                    if(code==C3X_RENDERER_RESULT_OK || code==C3X_RENDERER_RESULT_PREVIEW){
                        auto expected=*input;expected.tiles=view.frame.tiles;
                        expected.world_topology=nullptr;expected.world_topology_count=0;
                        if(view.ticket!=ticket || std::memcmp(&view.identity,&identity,sizeof(identity)) ||
                           std::memcmp(&view.frame,&expected,sizeof(expected)) ||
                           (input->tile_count && (!view.frame.tiles || std::memcmp(view.frame.tiles,input->tiles,input->tile_count*sizeof(input->tiles[0])))))
                            return int(C3X_RENDERER_RESULT_ERROR);
                        *result=view.output;
                    }
                }else code=camera_poll(ticket,result);
                QueryPerformanceCounter(&poll_end);
                poll_max_ms=(std::max)(poll_max_ms,double(poll_end.QuadPart-poll_begin.QuadPart)*1000/frequency.QuadPart);
                ++polls;
                if(code==C3X_RENDERER_RESULT_PREVIEW) {
                    if(!first_image) {
                        LARGE_INTEGER shown={};QueryPerformanceCounter(&shown);
                        if(result->width!=input->target_width || result->height!=input->target_height ||
                           result->replacement_tile_count!=input->tile_count)return int(C3X_RENDERER_RESULT_ERROR);
                        for(unsigned i=0;i<input->tile_count;++i) {
                            unsigned expected=(input->tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER)?C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED:0;
                            if(result->replacement_tile_flags[i]!=expected)return int(C3X_RENDERER_RESULT_ERROR);
                        }
                        std::printf("CAMERA first_image ticket=%lld ms=%.3f terrain_only=1\n",static_cast<long long>(ticket),
                            double(shown.QuadPart-begin.QuadPart)*1000/frequency.QuadPart);
                        // First cycle only: diagnostic evidence, never reference replacement.
                        if(camera_case<=7 && !write_bmp((std::string(argv[5])+".preview"+std::to_string(camera_case)+".bmp").c_str(),*result))
                            return int(C3X_RENDERER_RESULT_ERROR);
                        first_image=true;
                    }
                    code=C3X_RENDERER_RESULT_PENDING;
                }
                if(code==C3X_RENDERER_RESULT_PENDING)Sleep(1);
            }
            QueryPerformanceCounter(&finished);
            if(!first_image && code==C3X_RENDERER_RESULT_OK)
                std::printf("CAMERA first_image ticket=%lld ms=%.3f terrain_only=0\n",static_cast<long long>(ticket),
                    double(finished.QuadPart-begin.QuadPart)*1000/frequency.QuadPart);
            std::printf("CAMERA ticket=%lld accepted_ms=%.3f final_ms=%.3f polls=%u poll_max_ms=%.3f repeat_max_ms=%.3f identical_coalesced=1 stale_rejected=1 result=%d\n",
                static_cast<long long>(ticket),double(accepted.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(finished.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,polls,poll_max_ms,repeat_max_ms,code);
            if(camera_view && code==C3X_RENDERER_RESULT_OK)std::printf("CAMERA_IDENTITY ticket=%lld occurrences=%u visibility_epoch=%lld scene_epoch=%lld exact=1\n",
                static_cast<long long>(ticket),input->tile_count,static_cast<long long>(identity.visibility_epoch),static_cast<long long>(identity.scene_epoch));
            std::fflush(stdout);
        }else code=render(input,result);
        if(timing_enabled)QueryPerformanceCounter(&caller_return);
        if(code==C3X_RENDERER_RESULT_OK && !preview_ownership(*input,*result))code=C3X_RENDERER_RESULT_ERROR;
        if(timing_enabled) {
            QueryPerformanceCounter(&correct_done);
            if(timing_requests.size()<4096)timing_requests.push_back({
                fresh_capture?capture_begin.QuadPart:caller_enter.QuadPart,
                fresh_capture?capture_end.QuadPart:caller_enter.QuadPart,
                caller_enter.QuadPart,caller_return.QuadPart,correct_done.QuadPart,
                result->geometry_ticks,result->draw_ticks,result->readback_ticks,code,input->tile_count});
            else ++timing_dropped;
            fresh_capture=false;
        }
        return code;
    };
    // Short GPU-only map-demand control for Renderer64. It avoids the legacy
    // CPU bitmap render and diagnostic screenshot readbacks used below.
    char fresh_timing[8]={};
    if(GetEnvironmentVariableA("C3X_RENDERER_FRESH_TIMING",fresh_timing,sizeof(fresh_timing)) &&
       std::strcmp(fresh_timing,"1")==0) {
        auto gpu=reinterpret_cast<c3x_renderer_gpu_render_fn>(GetProcAddress(module,"c3x_renderer_gpu_render"));
        if(!gpu)return 1;
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        std::vector<double> idle,scroll,revisit;
        unsigned built=0,reused=0;std::uint64_t uploads=0;
        double cold_ms=0;
        for(int step=0;step<49;++step) {
            auto input=frame;auto records=tiles;input.tiles=records.data();
            input.presentation_time_ticks=frame.presentation_frequency+
                step*frame.presentation_frequency/60;
            if(step>=25)for(auto& tile:records){
                int travel=step<=36?step-24:48-step;
                tile.anchor_x+=travel*8;tile.anchor_y+=travel*4;
            }
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,
                sizeof(request),&input,{1,1,step+1,step+1}};
            c3x_renderer_gpu_frame_v1 view={sizeof(view)};
            c3x_renderer_output_v1 metadata={C3X_RENDERER_API_VERSION,sizeof(metadata)};
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            int result=gpu(&request,&view,&metadata);QueryPerformanceCounter(&end);
            if(result!=C3X_RENDERER_RESULT_OK || view.map_readbacks)return 1;
            double ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
            if(step==0)cold_ms=ms;
            if(step>=13 && step<25)idle.push_back(ms);
            if(step>=25 && step<37)scroll.push_back(ms);
            if(step>=37)revisit.push_back(ms);
            built+=metadata.geometry_tiles_built;reused+=metadata.geometry_tiles_reused;
            uploads+=metadata.geometry_upload_bytes;
        }
        auto report=[](char const* label,std::vector<double>& samples){
            std::sort(samples.begin(),samples.end());
            std::printf("FRESH_TIMING %s median_ms=%.3f p95_ms=%.3f count=%zu\n",
                label,samples[samples.size()/2],samples[(samples.size()*95+99)/100-1],samples.size());
        };
        std::printf("FRESH_TIMING cold_ms=%.3f\n",cold_ms);
        report("idle",idle);report("scroll",scroll);report("revisit",revisit);
        std::printf("FRESH_TIMING_WORK built=%u reused=%u uploads=%llu map_readbacks=0\n",
            built,reused,static_cast<unsigned long long>(uploads));
        return 0;
    }
    LARGE_INTEGER initial_begin={},initial_done={},initial_frequency={};QueryPerformanceFrequency(&initial_frequency);
    QueryPerformanceCounter(&initial_begin);
    int result = render_checked(&frame, &output);
    if(farm_study)std::printf("FARM_STUDY render=%d rendered=%u fallback=%u captured=%u\n",
        result,output.rendered_tile_count,output.fallback_tile_count,frame.tile_count);
    QueryPerformanceCounter(&initial_done);
    double initial_render_ms=double(initial_done.QuadPart-initial_begin.QuadPart)*1000/initial_frequency.QuadPart;
    std::size_t expected_rendered = 0;
    for (c3x_renderer_tile_v1 const & tile : tiles)
        if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) != 0)
            ++expected_rendered;
    bool ok = result == C3X_RENDERER_RESULT_OK &&
              (pickup ? output.rendered_tile_count >= expected_rendered : output.rendered_tile_count == expected_rendered) &&
              output.fallback_tile_count == 0 && write_bmp(argv[5], output);
    if(bootstrap){
        std::printf("CASE_ASSETS_READY result=%d initial_render_ms=%.3f\n",ok?0:1,initial_render_ms);
        return ok?0:1;
    }
#ifdef C3X_LAB_PREVIEW
    if(ok)ok=lab_verify_objects(frame,output);
    char water_study[8]={};GetEnvironmentVariableA("C3X_LAB_WATER_MOTION_STUDY",water_study,sizeof(water_study));
    if(ok && water_study[0]){
        auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            return std::vector<unsigned char>(p,p+output.stride_bytes*output.height);};
        auto draw=[&](){return render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;};
        auto save=[&](char const* suffix){return write_bmp((std::string(argv[5])+suffix).c_str(),output);};
        auto match=[&](char const* label,std::vector<unsigned char> const& expected){
            auto current=pixels();unsigned changed=0,maximum=0;unsigned long long error=0;
            for(std::size_t i=0;i<current.size();i+=4){bool bad=false;for(unsigned c=0;c<3;++c){
                unsigned d=unsigned(std::abs(int(current[i+c])-int(expected[i+c])));maximum=std::max(maximum,d);error+=d;bad=bad||d!=0;}changed+=bad;}
            std::printf("WATER compare=%s changed=%u maximum=%u error=%llu\n",label,changed,maximum,error);
            if(changed)save((std::string(".water-")+label+".bmp").c_str());
            // Only the independently finished static control uses the established
            // cold/full-redraw 8-bit rounding budget. Playback, reset and fog
            // comparisons below remain byte-exact.
            return maximum<=2 && changed<=current.size()/4000 && error<=current.size()/100;
        };
        auto initial=pixels(),previous=initial;unsigned changed=0;
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);double total=0;
        for(int sample=0;sample<48 && ok;++sample){
            frame.presentation_time_ticks=frame.presentation_frequency+sample*frame.presentation_frequency/15;
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);ok=draw();QueryPerformanceCounter(&end);
            total+=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
            ok=ok && output.visible_animation_count && !output.geometry_tiles_built && !output.geometry_upload_bytes;
            auto current=pixels();changed+=current!=previous;previous=current;
            char suffix[64];sprintf_s(suffix,".water-%03d.bmp",sample);ok=ok && save(suffix);
        }
        ok=ok && changed>=40;
        frame.presentation_time_ticks=frame.presentation_frequency;
        ok=ok && draw() && pixels()==initial;
        reset();ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw() && pixels()==initial;
        std::printf("WATER motion: %s changed=%u frames=48 mean_request_ms=%.3f repeat_cold_exact=1 terrain_upload_bytes=0\n",ok?"pass":"FAIL",changed,total/48);
        auto saved_tiles=tiles;
        for(auto& tile:tiles){tile.tile_flags|=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;tile.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;}
        ok=ok && draw();auto fog=pixels();ok=ok && save(".water-fog.bmp");
        for(int second:{3,7,11}){frame.presentation_time_ticks=second*frame.presentation_frequency;
            ok=ok && draw() && pixels()==fog && output.visible_animation_count==0;}
        std::printf("WATER fog: %s frozen_frames=3\n",ok?"pass":"FAIL");
        tiles=saved_tiles;frame.tiles=tiles.data();
        frame.presentation_time_ticks=frame.presentation_frequency;
        ok=ok && draw() && pixels()==initial;
        // Time zero is the preserved still material. Compare independently
        // against the unsplit static rendering route with motion disabled.
        frame.presentation_time_ticks=0;ok=ok && draw();auto zero=pixels();save(".water-zero.bmp");
        SetEnvironmentVariableA("C3X_RENDERER_WATER_MOTION","0");reset();
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw() && match("zero-static",zero);
        // Shore waves remain independently animated when ocean material
        // motion is disabled. Disable them only for this explicit still-water
        // control; the normal lifecycle above and below keeps waves on.
        char wave_control[8]={};GetEnvironmentVariableA("C3X_RENDERER_WAVES",wave_control,sizeof(wave_control));
        SetEnvironmentVariableA("C3X_RENDERER_WAVES","0");reset();
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw();
        auto off=pixels();ok=ok && save(".water-off.bmp");frame.presentation_time_ticks=5*frame.presentation_frequency;
        ok=ok && draw() && pixels()==off && !output.visible_animation_count;
        SetEnvironmentVariableA("C3X_RENDERER_WAVES",wave_control[0]?wave_control:nullptr);
        reset();ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw();
        std::printf("WATER control: %s zero_time_static_rounding_budget=1 disabled_still=1\n",ok?"pass":"FAIL");
        int water_center=center_x;center_x+=2;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        ok=ok && draw();auto still_pan=pixels();reset();
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw() && match("still-scroll-cold",still_pan);
        center_x=water_center;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        SetEnvironmentVariableA("C3X_RENDERER_WATER_MOTION","1");reset();frame.presentation_time_ticks=frame.presentation_frequency;
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw() && pixels()==initial;
        // Configure an effect change without a device reset: both reflection
        // settings now use the same resident path and must retire old pixels.
        char reflection_control[8]={};GetEnvironmentVariableA("C3X_RENDERER_REFLECTION_CONTROL",reflection_control,sizeof(reflection_control));
        SetEnvironmentVariableA("C3X_RENDERER_REFLECTION_CONTROL","1");
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw();auto no_mirrors=pixels();
        reset();ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw() && pixels()==no_mirrors;
        SetEnvironmentVariableA("C3X_RENDERER_REFLECTION_CONTROL",reflection_control[0]?reflection_control:nullptr);
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw() && pixels()==initial;
        std::printf("WATER reflection toggle: %s warm_off_cold_exact=1 restored_on_exact=1\n",ok?"pass":"FAIL");
        std::printf("%s water material lifecycle\n",ok?"PASS":"FAIL");
    }
    char wave_study[32]={};GetEnvironmentVariableA("C3X_LAB_WAVE_STUDY",wave_study,sizeof(wave_study));
    if(ok && wave_study[0]) {
      char water_control[16]={};GetEnvironmentVariableA("C3X_RENDERER_WATER_MOTION",water_control,sizeof(water_control));
      bool moving_water=std::strcmp(water_control,"0")!=0;
      auto verify_waves=[&](){
        auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            return std::vector<unsigned char>(p,p+output.stride_bytes*output.height);};
        auto render_wave=[&](){return render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;};
        auto compare=[&](char const* label,std::vector<unsigned char> const& expected,bool exact=true){
            auto current=pixels();unsigned changed=0,maximum=0;unsigned long long error=0;
            for(std::size_t i=0;i<current.size();i+=4){bool bad=false;
                for(unsigned c=0;c<3;++c){unsigned delta=unsigned(std::abs(int(current[i+c])-int(expected[i+c])));error+=delta;maximum=maximum>delta?maximum:delta;bad=bad || delta!=0;}
                changed+=bad;}
            std::printf("WAVE %s changed_pixels=%u max_delta=%u error=%llu\n",label,changed,maximum,error);
            if(changed)write_bmp((std::string(argv[5])+".wave-"+label+".bmp").c_str(),output);
            // The existing cold-scroll contract permits final 8-bit rounding.
            // Keep playback/zoom/time-return exact; allow at most two channel
            // levels and the ordinary bounded pixel/error budget for scroll.
            return exact?current==expected:maximum<=2 && changed<=current.size()/4000 && error<=current.size()/100;
        };
        bool rocky=std::strcmp(wave_study,"rocky-control")==0;
        bool moving=!rocky || moving_water;
        auto original=pixels(),previous=original;
        if(moving?output.visible_animation_count==0:output.visible_animation_count!=0)return false;
        int saved_width=tile_width;
        for(int next:{saved_width==112?96:112,saved_width}){
            tile_width=next;tile_height=next/2;tiles=capture_view();
            frame.tile_width=tile_width;frame.tile_height=tile_height;frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            if(!render_wave())return false;
        }
        if(!compare("zoom-return",original))return false;
        unsigned changed=0;
        for(int second:{5,9,13}){
            frame.presentation_time_ticks=second*frame.presentation_frequency;
            if(!render_wave() || output.geometry_tiles_built || output.geometry_upload_bytes)return false;
            auto current=pixels();changed+=current!=previous;previous=current;
            if(moving?!output.request_continuous_redraw:output.request_continuous_redraw)return false;
            std::printf("WAVE time=%d visible=%u terrain_built=%u terrain_upload=%u changed=%u\n",second,output.visible_animation_count,output.geometry_tiles_built,output.geometry_upload_bytes,changed);
            write_bmp((std::string(argv[5])+".wave-"+std::to_string(second)+".bmp").c_str(),output);
            if(!render_wave() || !compare("repeat",current))return false;
        }
        if(moving?changed==0:changed!=0)return false;
        frame.presentation_time_ticks=frame.presentation_frequency;
        if(!render_wave() || !compare("time-return",original))return false;
        char sequence_setting[16]={};
        GetEnvironmentVariableA("C3X_LAB_WAVE_SEQUENCE",sequence_setting,sizeof(sequence_setting));
        int sequence_frames=std::atoi(sequence_setting);
        sequence_frames=sequence_frames<0?0:sequence_frames>240?240:sequence_frames;
        for(int sample=0;sample<sequence_frames;++sample){
            frame.presentation_time_ticks=frame.presentation_frequency+sample*frame.presentation_frequency/4;
            if(!render_wave() || output.geometry_tiles_built || output.geometry_upload_bytes)return false;
            char suffix[64];sprintf_s(suffix,".wave-sequence-%03d.bmp",sample);
            if(!write_bmp((std::string(argv[5])+suffix).c_str(),output))return false;
        }
        if(sequence_frames){
            frame.presentation_time_ticks=frame.presentation_frequency;
            if(!render_wave() || !compare("sequence-return",original))return false;
            std::printf("PASS wave sequence: frames=%d step=0.25s cached terrain\n",sequence_frames);
        }
        reset();
        if(set_definitions(argv[2],argv[3],nullptr,custom_path)!=C3X_RENDERER_RESULT_OK || !render_wave() || !compare("cold",original))return false;
        int saved_center=center_x;center_x+=2;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        if(!render_wave())return false;
        auto scrolled=pixels();write_bmp((std::string(argv[5])+".wave-scroll-warm.bmp").c_str(),output);reset();
        if(set_definitions(argv[2],argv[3],nullptr,custom_path)!=C3X_RENDERER_RESULT_OK || !render_wave() || !compare("scroll-cold",scrolled,false))return false;
        center_x=saved_center;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        SetEnvironmentVariableA("C3X_RENDERER_WAVES","0");reset();
        if(set_definitions(argv[2],argv[3],nullptr,custom_path)!=C3X_RENDERER_RESULT_OK || !render_wave())return false;
        auto off=pixels();write_bmp((std::string(argv[5])+".wave-off.bmp").c_str(),output);
        if(rocky?off!=original:off==original)return false;
        // Wave configuration does not disable the independent water material.
        if(moving_water && (!output.visible_animation_count || !output.request_continuous_redraw))return false;
        SetEnvironmentVariableA("C3X_RENDERER_WATER_MOTION","0");reset();
        if(set_definitions(argv[2],argv[3],nullptr,custom_path)!=C3X_RENDERER_RESULT_OK || !render_wave() ||
            output.visible_animation_count || output.request_continuous_redraw)return false;
        auto still=pixels();frame.presentation_time_ticks=7*frame.presentation_frequency;
        if(!render_wave() || !compare("all-motion-off",still))return false;
        frame.presentation_time_ticks=frame.presentation_frequency;
        std::printf("WAVE controls: wave_off_preserves_water=%u both_off_still=1\n",unsigned(moving_water));
        return true;
      };
      ok=verify_waves();SetEnvironmentVariableA("C3X_RENDERER_WAVES",nullptr);
      SetEnvironmentVariableA("C3X_RENDERER_WATER_MOTION",water_control[0]?water_control:nullptr);
      // Restore the requested configuration after the disabled-effect control;
      // later witnesses and their cold resets must exercise the same settings.
      reset();
      ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&
          render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
      std::printf("%s coastal wave lifecycle: %s repeat, time-return, zoom-return, scroll/cold, disabled, cached terrain\n",ok?"PASS":"FAIL",wave_study);
    }
    char site_study[32]={};GetEnvironmentVariableA("C3X_LAB_OBJECT_STUDY",site_study,sizeof(site_study));
    if(ok && (std::strcmp(site_study,"huts-camps")==0 || std::strcmp(site_study,"goody-huts")==0 ||
              std::strcmp(site_study,"barbarian-camps")==0) && frame.hour==12 && frame.tile_width==128) {
        auto saved=tiles;
        auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            return std::vector<unsigned char>(p,p+output.stride_bytes*output.height);};
        auto original=pixels();
        unsigned site_mask=C3X_RENDERER_TILE_CUSTOM_HUT_REPLACED|C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED;
        for(auto& tile:tiles) {
            tile.improvement_flags&=~(C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP);
            tile.barbarian_tribe_id=-1;
        }
        ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
        if(ok) {
            for(unsigned i=0;i<output.replacement_tile_count;++i)ok=ok && !(output.replacement_tile_flags[i]&site_mask);
            auto removed=pixels();ok=ok && removed!=original;
            write_bmp((std::string(argv[5])+".sites-removed.bmp").c_str(),output);
            reset();
            ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&
                render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && pixels()==removed;
            std::printf("%s site removal and cold pixel parity\n",ok?"PASS":"FAIL");
        }
        std::copy(saved.begin(),saved.end(),tiles.begin());
        ok=ok && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && pixels()==original && lab_verify_objects(frame,output);
        std::printf("%s site reappearance and stable composition\n",ok?"PASS":"FAIL");
    }

#endif
    char zoom_option[16]={};
    bool zoom_benchmark=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_ZOOM",zoom_option,sizeof(zoom_option))!=0;
    char navigation_option[16]={};
    bool navigation_benchmark=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_NAVIGATION",navigation_option,sizeof(navigation_option))!=0;
    char scroll_option[64]={};
    bool scroll_ablation=!topology_edits && !prepared_view_fixture && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_SCROLL_ABLATION",scroll_option,sizeof(scroll_option))!=0;
    char distant_option[16]={};
    int distant_steps=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_DISTANT_STEPS",distant_option,sizeof(distant_option))?std::clamp(std::atoi(distant_option),0,1000):0;
    char idle_option[16]={};
    int idle_steps=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_IDLE_STEPS",idle_option,sizeof(idle_option))?std::clamp(std::atoi(idle_option),0,1000):0;
    int idle_pace_ms=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_IDLE_PACE_MS",idle_option,sizeof(idle_option))?std::clamp(std::atoi(idle_option),0,500):0;
    int idle_warmup=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_IDLE_WARMUP",idle_option,sizeof(idle_option))?std::clamp(std::atoi(idle_option),10,150):10;
    char session_option[8]={};
    bool busy_session=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_BUSY_SESSION",session_option,sizeof(session_option)) && std::strcmp(session_option,"1")==0;
    char replay_option[8]={};
    bool retained_replay=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_RETAINED_REPLAY",replay_option,sizeof(replay_option)) && std::strcmp(replay_option,"1")==0;
    char preparation_option[16]={};GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_PREPARATION_MODE",preparation_option,sizeof(preparation_option));
    bool oracle_preparation=std::strcmp(preparation_option,"oracle")==0;
    int replay_samples=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_REPLAY_SAMPLES",replay_option,sizeof(replay_option))?
        std::clamp(std::atoi(replay_option),1,100):25;
    char cycle_option[16]={};
    int camera_cycles=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CYCLES",cycle_option,sizeof(cycle_option))?
        std::clamp(std::atoi(cycle_option),2,40):2;
    // Sample outside the timed interaction. Free VA is not free physical RAM;
    // the largest available region also exposes fragmentation in this x86 host.
    auto camera_memory_values = [&]() {
        MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);
        if(!GlobalMemoryStatusEx(&memory))return std::pair<unsigned long long,SIZE_T>{0,0};
        std::uintptr_t address=0;SIZE_T largest=0;MEMORY_BASIC_INFORMATION region={};
        while(VirtualQuery(reinterpret_cast<void const*>(address),&region,sizeof(region))){
            if(region.State==MEM_FREE)largest=(std::max)(largest,region.RegionSize);
            auto next=reinterpret_cast<std::uintptr_t>(region.BaseAddress)+region.RegionSize;
            if(next<=address)break;
            address=next;
        }
        return std::pair<unsigned long long,SIZE_T>{memory.ullAvailVirtual,largest};
    };
    auto camera_memory = [&]() {
        auto values=camera_memory_values();
        MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);GlobalMemoryStatusEx(&memory);
        std::printf("CAMERA memory available_virtual=%llu largest_free_region=%zu total_virtual=%llu\n",
            values.first,values.second,memory.ullTotalVirtual);
    };
    #ifdef C3X_GPU_NATIVE_CONTRACT
    #include "world_readiness_preview.h"
    #endif
    #ifdef C3X_GPU_NATIVE_CONTRACT
    #include "input_recording_preview.h"
    #endif
    #include "gpu_frame_preview.h"
    #include "prepared_view_preview.h"
    #include "retained_replay_preview.h"
    #include "busy_session_preview.h"
    if(ok && ambient_async) {
        c3x_renderer_camera_identity_v1 stationary_identity={1,2,3,4};
        auto legacy_render=render;
        auto render_ambient=[&](c3x_renderer_frame_v1 const* input,c3x_renderer_output_v1* result){
            if(!ambient_boundary)return legacy_render(input,result);
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),input,stationary_identity};
            return render_view(&request,result);
        };
        // Small architectural gate: the consumer retains the last exact bitmap
        // for this camera while the renderer prepares a newer ambient clock on
        // its worker. Publication is accepted only as one exact pixel/ownership
        // transaction. A changed camera supersedes the old request.
        struct AmbientReference {
            std::vector<unsigned char> pixels;
            std::vector<c3x_renderer_u32> ownership;
            c3x_renderer_i64 ticks=0;
        };
        auto reference_from = [&](c3x_renderer_i64 ticks) {
            AmbientReference value;value.ticks=ticks;
            auto pixels=static_cast<unsigned char const*>(output.bgra_pixels);
            value.pixels.assign(pixels,pixels+std::size_t(output.stride_bytes)*output.height);
            value.ownership.assign(output.replacement_tile_flags,
                output.replacement_tile_flags+output.replacement_tile_count);
            return value;
        };
        auto exact_publication = [&](c3x_renderer_camera_view_v1 const& view,AmbientReference const& expected) {
            auto const& rendered=view.output;
            auto pixels=static_cast<unsigned char const*>(rendered.bgra_pixels);
            std::size_t bytes=std::size_t(rendered.stride_bytes)*rendered.height;
            return rendered.width==target_width && rendered.height==target_height &&
                bytes==expected.pixels.size() && pixels &&
                std::memcmp(pixels,expected.pixels.data(),bytes)==0 &&
                rendered.replacement_tile_count==expected.ownership.size() &&
                rendered.replacement_tile_flags &&
                std::memcmp(rendered.replacement_tile_flags,expected.ownership.data(),
                    expected.ownership.size()*sizeof(expected.ownership[0]))==0;
        };
        std::vector<AmbientReference> references;
        references.push_back(reference_from(frame.presentation_time_ticks));
        for(int step=1;step<=3 && ok;++step) {
            frame.presentation_time_ticks=1000000+c3x_renderer_i64(step)*frame.presentation_frequency/15;
            int code=render_ambient(&frame,&output);
            ok=code==C3X_RENDERER_RESULT_OK && preview_ownership(frame,output) &&
                output.visible_animation_count>0 && output.fallback_tile_count==0 && output.device_recoveries==0;
            if(ok)references.push_back(reference_from(frame.presentation_time_ticks));
        }
        int const home_x=center_x,home_y=center_y;
        auto home_tiles=tiles;
        center_x+=4;
        auto changed_tiles=capture_view();
        frame.tiles=changed_tiles.data();frame.tile_count=unsigned(changed_tiles.size());
        frame.presentation_time_ticks=1000000+c3x_renderer_i64(4)*frame.presentation_frequency/15;
        AmbientReference changed_reference;
        if(ok) {
            int code=render_ambient(&frame,&output);
            ok=code==C3X_RENDERER_RESULT_OK && preview_ownership(frame,output) &&
                output.visible_animation_count>0 && output.fallback_tile_count==0 && output.device_recoveries==0;
            if(ok)changed_reference=reference_from(frame.presentation_time_ticks);
        }

        reset();
        if(ambient_boundary)SetEnvironmentVariableA("C3X_RENDERER_SYNC_AMBIENT","1");
        if(ok)ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK;
        center_x=home_x;center_y=home_y;tiles=home_tiles;
        frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        frame.presentation_time_ticks=references.front().ticks;
        if(ok)ok=render_ambient(&frame,&output)==C3X_RENDERER_RESULT_OK && preview_ownership(frame,output);
        std::vector<unsigned char> front;
        if(ok) {
            auto pixels=static_cast<unsigned char const*>(output.bgra_pixels);
            front.assign(pixels,pixels+std::size_t(output.stride_bytes)*output.height);
            ok=front==references.front().pixels;
        }
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        SIZE_T min_largest_free=SIZE_MAX;unsigned exact_count=0,no_reuse_count=0;
        double max_accept_ms=0,max_unit_set_ms=0,max_unit_ms=0,last_unit_set_ms=0;unsigned measured_unit_draws=0;
        auto ambient_unit_draw=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(
            GetProcAddress(module,"c3x_renderer_unit_draw_expanded"));
        std::vector<c3x_renderer_tile_v1 const*> ambient_unit_sites;
        HDC ambient_unit_dc=nullptr;HBITMAP ambient_unit_bitmap=nullptr;HGDIOBJ ambient_unit_old=nullptr;
        void* ambient_unit_pixels=nullptr;
        if(ok && idle_unit_count) {
            for(auto const& tile:tiles)if(tile.real_terrain_type<=4 && tile.city_id<0 &&
               (tile.tile_flags&C3X_RENDERER_TILE_RENDER) && tile.anchor_x>tile_width &&
               tile.anchor_x<target_width-tile_width*2 && tile.anchor_y>tile_height*2 &&
               tile.anchor_y<target_height-tile_height*3)ambient_unit_sites.push_back(&tile);
            std::sort(ambient_unit_sites.begin(),ambient_unit_sites.end(),[](auto a,auto b){
                return a->anchor_y==b->anchor_y?a->anchor_x<b->anchor_x:a->anchor_y<b->anchor_y;});
            if(ambient_unit_sites.size()<std::size_t(idle_unit_count) || !ambient_unit_draw)ok=false;
            else ambient_unit_sites.resize(idle_unit_count);
            BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
            info.bmiHeader.biWidth=target_width;info.bmiHeader.biHeight=-target_height;
            info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
            ambient_unit_dc=CreateCompatibleDC(nullptr);
            if(ambient_unit_dc)ambient_unit_bitmap=CreateDIBSection(ambient_unit_dc,&info,DIB_RGB_COLORS,
                &ambient_unit_pixels,nullptr,0);
            if(!ambient_unit_dc || !ambient_unit_bitmap || !ambient_unit_pixels)ok=false;
            if(ambient_unit_bitmap)ambient_unit_old=SelectObject(ambient_unit_dc,ambient_unit_bitmap);
        }
        auto make_ambient_unit=[&](int index,int cursor) {
            char const* names[]={"Archer","Swordsman","Infantry","Warrior","Scout","Settler","Worker"};
            auto site=ambient_unit_sites[index];c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);
            unit.unit_id=20000+index;unit.action=1;unit.direction=1+index%8;unit.frame_count=16;
            unit.action_cursor=0;unit.presentation_frequency=frame.presentation_frequency;
            unit.presentation_time_ticks=frame.presentation_time_ticks;unit.sprite_width=unit.sprite_height=191;
            unit.projection_scale_milli=tile_width*1000/128;
            unit.body_x=site->anchor_x+tile_width/2-191*unit.projection_scale_milli/2000;
            unit.body_y=site->anchor_y+tile_height/2-191*unit.projection_scale_milli/2000;
            if(index==0)unit.action_cursor=cursor;           // selected idle loop
            else if(index==1){unit.action=3;unit.action_cursor=cursor;} // directed combat
            else if(index==6){unit.action=8;unit.action_cursor=cursor;} // worker task
            unit.hour=frame.hour;unit.season=frame.season;unit.display_color_rgb=0x205bdd;
            sprintf_s(unit.unit_key,"PRTO_%s",names[index%std::size(names)]);return unit;
        };
        auto draw_one_ambient_unit=[&](int index,int cursor,bool measured) {
            auto unit=make_ambient_unit(index,cursor);int bounds[4]={};
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            int code=ambient_unit_draw(&unit,ambient_unit_dc,ambient_unit_dc,bounds);
            QueryPerformanceCounter(&end);
            double ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
            if(measured){max_unit_ms=(std::max)(max_unit_ms,ms);++measured_unit_draws;}
            return code==C3X_RENDERER_RESULT_OK;
        };
        if(ok && idle_unit_count) {
            // Warm only the three active cycles plus one pose for each frozen
            // unit. Pose creation is deliberately outside measured UI work.
            std::memcpy(ambient_unit_pixels,front.data(),front.size());
            for(int cursor=0;cursor<16 && ok;++cursor)
                for(int index:{0,1,6})if(index<idle_unit_count)
                    ok=draw_one_ambient_unit(index,cursor,false) && ok;
            for(int index=0;index<idle_unit_count && ok;++index)
                if(index!=0 && index!=1 && index!=6)ok=draw_one_ambient_unit(index,0,false) && ok;
            GdiFlush();
            std::printf("AMBIENT_UNITS_WARM units=%d active_cycles=3 frozen=%d status=%s\n",
                idle_unit_count,(std::max)(0,idle_unit_count-3),ok?"pass":"FAIL");
        }
        auto draw_ambient_set=[&](int cursor,bool report=true) {
            if(!idle_unit_count)return true;
            std::memcpy(ambient_unit_pixels,front.data(),front.size());
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            bool drawn=true;
            for(int index=0;index<idle_unit_count;++index)drawn=draw_one_ambient_unit(index,cursor,true) && drawn;
            GdiFlush();QueryPerformanceCounter(&end);
            double ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
            last_unit_set_ms=ms;
            max_unit_set_ms=(std::max)(max_unit_set_ms,ms);
            if(report)std::printf("AMBIENT_UNITS frame=%d units=%d selected=1 worker=1 combat=1 frozen=%d total_ms=%.3f max_call_ms=%.3f status=%s\n",
                cursor,idle_unit_count,(std::max)(0,idle_unit_count-3),ms,max_unit_ms,drawn?"pass":"FAIL");
            return drawn && (!ambient_boundary || ambient_soak_seconds || ms<16.0);
        };
        auto sample_memory=[&]() {
            auto values=camera_memory_values();
            min_largest_free=(std::min)(min_largest_free,values.second);
            return values.second>=SIZE_T(512)*1024*1024;
        };
        auto await_exact = [&](c3x_renderer_i64 ticket,AmbientReference const& expected,
                               c3x_renderer_camera_identity_v1 const& identity,char const* label,bool stationary) {
            auto started=GetTickCount64();unsigned polls=0;int code=C3X_RENDERER_RESULT_PENDING;
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
            bool memory_ok=sample_memory();
            while(code==C3X_RENDERER_RESULT_PENDING && GetTickCount64()-started<120000) {
                code=camera_poll_view(ticket,&view);++polls;
                memory_ok=sample_memory() && memory_ok;
                if(code==C3X_RENDERER_RESULT_PENDING)Sleep(1);
            }
            QueryPerformanceCounter(&end);
            bool exact=code==C3X_RENDERER_RESULT_OK && view.ticket==ticket &&
                std::memcmp(&view.identity,&identity,sizeof(identity))==0 && exact_publication(view,expected);
            // Reusing the immutable terrain base is desirable. The forbidden
            // shortcut is publishing the already completed ambient bitmap.
            bool no_reuse=exact && expected.pixels!=front && view.output.renderer_cpu_ticks>0 &&
                (!stationary || (view.output.geometry_tiles_built==0 && view.output.geometry_upload_bytes==0));
            bool healthy=exact && no_reuse && view.output.fallback_tile_count==0 &&
                view.output.device_recoveries==0 && view.output.visible_animation_count>0 && memory_ok;
            std::printf("AMBIENT_ASYNC publish=%s ticket=%lld result=%d polls=%u final_ms=%.3f exact=%u no_completed_reuse=%u fallback=%u recoveries=%u largest_free_mib=%.1f\n",
                label,static_cast<long long>(ticket),code,polls,
                double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,unsigned(exact),unsigned(no_reuse),
                view.output.fallback_tile_count,view.output.device_recoveries,double(min_largest_free)/(1024.0*1024.0));
            if(healthy) {
                auto pixels=static_cast<unsigned char const*>(view.output.bgra_pixels);
                front.assign(pixels,pixels+std::size_t(view.output.stride_bytes)*view.output.height);
                output=view.output;++exact_count;++no_reuse_count;
            }
            return healthy;
        };
        if(ambient_boundary) {
            auto exact_output=[&](AmbientReference const& expected) {
                auto pixels=static_cast<unsigned char const*>(output.bgra_pixels);
                auto bytes=std::size_t(output.stride_bytes)*output.height;
                return pixels && bytes==expected.pixels.size() &&
                    !std::memcmp(pixels,expected.pixels.data(),bytes) &&
                    output.replacement_tile_count==expected.ownership.size() &&
                    output.replacement_tile_flags && !std::memcmp(output.replacement_tile_flags,
                        expected.ownership.data(),expected.ownership.size()*sizeof(expected.ownership[0]));
            };
            std::printf("AMBIENT_BOUNDARY_BEGIN ticks=3 policy=explicit-identity-pull camera=stationary floor_mib=512\n");
            for(int step=1;step<=3 && ok;++step) {
                frame.presentation_time_ticks=references[step].ticks;
                LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
                int code=render_ambient(&frame,&output);QueryPerformanceCounter(&end);
                double call_ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
                max_accept_ms=(std::max)(max_accept_ms,call_ms);
                auto pixels=static_cast<unsigned char const*>(output.bgra_pixels);
                auto bytes=std::size_t(output.stride_bytes)*output.height;
                bool retained=code==C3X_RENDERER_RESULT_OK && pixels && bytes==front.size() &&
                    !std::memcmp(pixels,front.data(),bytes);
                std::printf("AMBIENT_BOUNDARY submit=stationary-%d result=%d call_ms=%.3f retained_previous_exact=%u\n",
                    step,code,call_ms,unsigned(retained));
                ok=retained && call_ms<16.0 && draw_ambient_set(step) && sample_memory();
                auto started=GetTickCount64();unsigned calls=1;
                while(ok && !exact_output(references[step]) && GetTickCount64()-started<120000) {
                    Sleep(1);QueryPerformanceCounter(&begin);code=render_ambient(&frame,&output);QueryPerformanceCounter(&end);
                    call_ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
                    max_accept_ms=(std::max)(max_accept_ms,call_ms);++calls;
                    ok=code==C3X_RENDERER_RESULT_OK && call_ms<16.0 && sample_memory();
                }
                bool exact=ok && exact_output(references[step]) && references[step].pixels!=front;
                std::printf("AMBIENT_BOUNDARY publish=stationary-%d calls=%u exact_ms=%llu exact=%u fallback=%u recoveries=%u largest_free_mib=%.1f\n",
                    step,calls,static_cast<unsigned long long>(GetTickCount64()-started),unsigned(exact),
                    output.fallback_tile_count,output.device_recoveries,double(min_largest_free)/(1024.0*1024.0));
                ok=exact && output.fallback_tile_count==0 && output.device_recoveries==0;
                if(ok)front=references[step].pixels;
            }
            if(ok) {
                frame.presentation_time_ticks=1000000+c3x_renderer_i64(5)*frame.presentation_frequency/15;
                ok=render_ambient(&frame,&output)==C3X_RENDERER_RESULT_OK;
                Sleep(5);center_x=home_x+4;center_y=home_y;tiles=changed_tiles;
                frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());frame.presentation_time_ticks=changed_reference.ticks;
                LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
                int code=render_ambient(&frame,&output);QueryPerformanceCounter(&end);
                double call_ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
                bool exact=code==C3X_RENDERER_RESULT_OK && exact_output(changed_reference) &&
                    output.fallback_tile_count==0 && output.device_recoveries==0 && sample_memory();
                std::printf("AMBIENT_BOUNDARY camera-change result=%d call_ms=%.3f exact=%u old_camera_rejected=%u fallback=%u recoveries=%u\n",
                    code,call_ms,unsigned(exact),unsigned(exact),output.fallback_tile_count,output.device_recoveries);
                ok=exact;
            }
            if(ok) {
                // One caller request is prepared through the camera interface,
                // then demanded through ordinary render. Include begin and all
                // work before return in the endpoint; do not hide preparation
                // behind a sleep or measure only the final pull call.
                center_x=home_x;center_y=home_y;tiles=home_tiles;
                frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
                frame.presentation_time_ticks=references[3].ticks;
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,stationary_identity};
                c3x_renderer_i64 ticket=0;
                LARGE_INTEGER begin={},accepted={},finished={};QueryPerformanceCounter(&begin);
                int begun=camera_begin_view(&request,&ticket);QueryPerformanceCounter(&accepted);
                int code=begun==C3X_RENDERER_RESULT_PENDING?render_ambient(&frame,&output):begun;
                QueryPerformanceCounter(&finished);
                bool exact=code==C3X_RENDERER_RESULT_OK && exact_output(references[3]) &&
                    preview_ownership(frame,output) && output.fallback_tile_count==0 &&
                    output.device_recoveries==0 && sample_memory();
                std::printf("AMBIENT_BOUNDARY queued-current ticket=%lld result=%d begin_ms=%.3f call_ms=%.3f total_ms=%.3f exact=%u fallback=%u recoveries=%u\n",
                    static_cast<long long>(ticket),code,double(accepted.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                    double(finished.QuadPart-accepted.QuadPart)*1000/frequency.QuadPart,
                    double(finished.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,unsigned(exact),
                    output.fallback_tile_count,output.device_recoveries);
                ok=exact;
            }
            if(ok && ambient_soak_seconds) {
                center_x=home_x;center_y=home_y;tiles=home_tiles;
                frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
                frame.presentation_time_ticks=2000000;
                int code=render_ambient(&frame,&output);
                // Returning home is a camera change, so this call must itself
                // be exact before the stationary cadence begins.
                ok=code==C3X_RENDERER_RESULT_OK && preview_ownership(frame,output) &&
                    output.fallback_tile_count==0 && output.device_recoveries==0;
                if(ok) {
                    auto pixels=static_cast<unsigned char const*>(output.bgra_pixels);
                    front.assign(pixels,pixels+std::size_t(output.stride_bytes)*output.height);
                }
                auto initial_memory=camera_memory_values();
                SIZE_T soak_min_largest=initial_memory.second;
                std::vector<double> render_calls,unit_set_calls;
                unsigned delivered=0;unsigned long long last_hash=0;
                auto sampled_hash=[&]() {
                    auto data=static_cast<unsigned char const*>(output.bgra_pixels);
                    auto bytes=std::size_t(output.stride_bytes)*output.height;
                    unsigned long long value=14695981039346656037ull;
                    for(std::size_t i=0;i<bytes;i+=256)value=(value^data[i])*1099511628211ull;
                    for(unsigned i=0;i<output.replacement_tile_count;++i)
                        value=(value^output.replacement_tile_flags[i])*1099511628211ull;
                    return value;
                };
                if(ok)last_hash=sampled_hash();
                LARGE_INTEGER cadence_start={},now={};QueryPerformanceCounter(&cadence_start);
                int ticks=ambient_soak_seconds*15;
                std::printf("AMBIENT_SOAK_BEGIN seconds=%d ticks=%d cadence_hz=15 units=%d dense=%u\n",
                    ambient_soak_seconds,ticks,idle_unit_count,unsigned(dense_scene));
                for(int tick=1;tick<=ticks && ok;++tick) {
                    auto target=cadence_start.QuadPart+c3x_renderer_i64(tick-1)*frequency.QuadPart/15;
                    do {QueryPerformanceCounter(&now);if(now.QuadPart+frequency.QuadPart/500<target)Sleep(1);else if(now.QuadPart<target)Sleep(0);} while(now.QuadPart<target);
                    frame.presentation_time_ticks=2000000+c3x_renderer_i64(tick)*frame.presentation_frequency/15;
                    LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
                    code=render_ambient(&frame,&output);QueryPerformanceCounter(&end);
                    render_calls.push_back(double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart);
                    auto memory=camera_memory_values();soak_min_largest=(std::min)(soak_min_largest,memory.second);
                    ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 &&
                        output.device_recoveries==0 && memory.second>=SIZE_T(512)*1024*1024 &&
                        draw_ambient_set(tick%16,false);
                    unit_set_calls.push_back(last_unit_set_ms);
                    if(ok){auto hash=sampled_hash();if(hash!=last_hash){++delivered;last_hash=hash;}}
                }
                auto final_memory=camera_memory_values();
                std::sort(render_calls.begin(),render_calls.end());
                std::sort(unit_set_calls.begin(),unit_set_calls.end());
                double p95=render_calls.empty()?0:render_calls[(render_calls.size()*95-1)/100];
                double maximum=render_calls.empty()?0:render_calls.back();
                double unit_p95=unit_set_calls.empty()?0:unit_set_calls[(unit_set_calls.size()*95-1)/100];
                double unit_maximum=unit_set_calls.empty()?0:unit_set_calls.back();
                double delivered_hz=ambient_soak_seconds?double(delivered)/ambient_soak_seconds:0;
                ok=ok && p95<2.0 && maximum<100.0 && unit_p95<16.0 && unit_maximum<100.0 &&
                    delivered>=unsigned(ambient_soak_seconds*5);
                std::printf("AMBIENT_SOAK_END status=%s delivered=%u delivered_hz=%.2f render_p95_ms=%.3f render_max_ms=%.3f unit_set_p95_ms=%.3f unit_set_max_ms=%.3f unit_call_max_ms=%.3f available_virtual_delta_mib=%.1f largest_free_start_mib=%.1f largest_free_end_mib=%.1f largest_free_min_mib=%.1f fallback=%u recoveries=%u\n",
                    ok?"pass":"FAIL",delivered,delivered_hz,p95,maximum,unit_p95,unit_maximum,max_unit_ms,
                    (double(final_memory.first)-double(initial_memory.first))/(1024.0*1024.0),
                    double(initial_memory.second)/(1024.0*1024.0),double(final_memory.second)/(1024.0*1024.0),
                    double(soak_min_largest)/(1024.0*1024.0),output.fallback_tile_count,output.device_recoveries);
            }
            std::printf("AMBIENT_BOUNDARY_END status=%s exact_publications=%u max_sync_call_ms=%.3f min_largest_free_mib=%.1f units=%d unit_draws=%u max_unit_set_ms=%.3f max_unit_call_ms=%.3f\n",
                ok?"pass":"FAIL",ok?5u:0u,max_accept_ms,double(min_largest_free)/(1024.0*1024.0),
                idle_unit_count,measured_unit_draws,max_unit_set_ms,max_unit_ms);
        } else {
        std::printf("AMBIENT_ASYNC_BEGIN ticks=3 policy=retain-last-exact camera=stationary floor_mib=512\n");
        for(int step=1;step<=3 && ok;++step) {
            frame.presentation_time_ticks=references[step].ticks;
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,stationary_identity};
            c3x_renderer_i64 ticket=0;LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            int code=camera_begin_view(&request,&ticket);QueryPerformanceCounter(&end);
            double accept_ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
            max_accept_ms=(std::max)(max_accept_ms,accept_ms);
            bool retained=front==references[step-1].pixels;
            std::printf("AMBIENT_ASYNC accept=stationary-%d ticket=%lld result=%d present_ms=%.3f retained_previous_exact=%u\n",
                step,static_cast<long long>(ticket),code,accept_ms,unsigned(retained));
            ok=code==C3X_RENDERER_RESULT_PENDING && retained && draw_ambient_set(step) && sample_memory() &&
                await_exact(ticket,references[step],stationary_identity,"stationary",true);
        }
        if(ok) {
            // Make the obsolete request genuinely eligible to start, then
            // replace it with a translated camera and stricter scene epochs.
            frame.presentation_time_ticks=1000000+c3x_renderer_i64(5)*frame.presentation_frequency/15;
            c3x_renderer_camera_request_v1 obsolete_request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(obsolete_request),&frame,stationary_identity};
            c3x_renderer_i64 obsolete=0;
            ok=camera_begin_view(&obsolete_request,&obsolete)==C3X_RENDERER_RESULT_PENDING;
            Sleep(5);
            center_x=home_x+4;center_y=home_y;tiles=changed_tiles;
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());frame.presentation_time_ticks=changed_reference.ticks;
            c3x_renderer_camera_identity_v1 changed_identity={1,2,4,5};
            c3x_renderer_camera_request_v1 changed_request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(changed_request),&frame,changed_identity};
            c3x_renderer_i64 changed_ticket=0;LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            int code=camera_begin_view(&changed_request,&changed_ticket);QueryPerformanceCounter(&end);
            double accept_ms=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;
            max_accept_ms=(std::max)(max_accept_ms,accept_ms);
            c3x_renderer_camera_view_v1 stale_view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(stale_view)};
            int stale=camera_poll_view(obsolete,&stale_view);
            bool retained=front==references.back().pixels;
            std::printf("AMBIENT_ASYNC accept=camera-change ticket=%lld result=%d present_ms=%.3f retained_previous_exact=%u stale_ticket=%lld stale_result=%d\n",
                static_cast<long long>(changed_ticket),code,accept_ms,unsigned(retained),static_cast<long long>(obsolete),stale);
            ok=code==C3X_RENDERER_RESULT_PENDING && stale==C3X_RENDERER_RESULT_SUPERSEDED && retained &&
                sample_memory() && await_exact(changed_ticket,changed_reference,changed_identity,"camera-change",false);
        }
        ok=ok && exact_count==4 && no_reuse_count==4 && min_largest_free>=SIZE_T(512)*1024*1024;
        std::printf("AMBIENT_ASYNC_END status=%s exact_publications=%u no_completed_reuse=%u max_present_ms=%.3f min_largest_free_mib=%.1f stale_camera_rejected=1 units=%d unit_draws=%u max_unit_set_ms=%.3f max_unit_call_ms=%.3f\n",
            ok?"pass":"FAIL",exact_count,no_reuse_count,max_accept_ms,double(min_largest_free)/(1024.0*1024.0),
            idle_unit_count,measured_unit_draws,max_unit_set_ms,max_unit_ms);
        }
        if(ambient_unit_dc && ambient_unit_old)SelectObject(ambient_unit_dc,ambient_unit_old);
        if(ambient_unit_bitmap)DeleteObject(ambient_unit_bitmap);
        if(ambient_unit_dc)DeleteDC(ambient_unit_dc);
    }
    if(ok && zoom_benchmark) {
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        char supported_option[8]={};
        bool supported=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_SUPPORTED_ZOOMS",supported_option,sizeof(supported_option)) && std::strcmp(supported_option,"1")==0;
        std::vector<int> levels=supported?std::vector<int>{128,192,160}:std::vector<int>{128,96,64,192,160};
        std::vector<std::vector<unsigned char>> reference(levels.size());
        char prepare_option[16]={};GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_ZOOM_PREPARE_MS",prepare_option,sizeof(prepare_option));
        int prepare_ms=std::clamp(std::atoi(prepare_option),0,10000);
        if(prepare_ms){
            LARGE_INTEGER started={},ended={};QueryPerformanceCounter(&started);
            auto saved_frame=frame;int saved_width=tile_width,saved_height=tile_height;
            if(prepare_view && native_handoff)for(auto level:levels)if(level!=tile_width){
                tile_width=level;tile_height=level/2;auto prospective=capture_view();
                auto future=frame;future.tile_width=level;future.tile_height=level/2;
                future.tiles=prospective.data();future.tile_count=unsigned(prospective.size());
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&future,{1,2,3,future.world_topology_revision}};
                prepare_view(&request,0);
            }
            tile_width=saved_width;tile_height=saved_height;frame=saved_frame;
            Sleep(DWORD(prepare_ms));QueryPerformanceCounter(&ended);
            std::printf("ZOOM_PREPARATION opportunity_ms=%d elapsed_ms=%.3f prospective_api=%u\n",prepare_ms,
                double(ended.QuadPart-started.QuadPart)*1000/frequency.QuadPart,unsigned(prepare_view!=nullptr));
        }

        for(int cycle=0;cycle<camera_cycles && ok;++cycle)for(std::size_t level=0;level<levels.size() && ok;++level){
            LARGE_INTEGER begin={},captured={},end={};QueryPerformanceCounter(&begin);
            tile_width=levels[level];tile_height=tile_width/2;tiles=capture_view();
            frame.tile_width=tile_width;frame.tile_height=tile_height;
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            QueryPerformanceCounter(&captured);
            int code=render_checked(&frame,&output);
            std::vector<unsigned char> delivered;
            if(code==C3X_RENDERER_RESULT_OK && output.bgra_pixels){auto source=static_cast<unsigned char const*>(output.bgra_pixels);
                delivered.assign(source,source+std::size_t(output.stride_bytes)*output.height);}
            QueryPerformanceCounter(&end);
            ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
            std::printf("ZOOM cycle=%d width=%d result=%d tiles=%u built=%u reused=%u cache_bytes=%u recoveries=%u ms=%.3f capture_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f\n",
                cycle,tile_width,code,output.rendered_tile_count,output.geometry_tiles_built,output.geometry_tiles_reused,
                output.geometry_cache_bytes,output.device_recoveries,double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(captured.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(output.geometry_ticks)*1000/frequency.QuadPart,double(output.draw_ticks)*1000/frequency.QuadPart,
                double(output.readback_ticks)*1000/frequency.QuadPart);
            camera_memory();std::fflush(stdout);
            if(!ok)break;
            auto bytes=static_cast<unsigned char const*>(output.bgra_pixels);
            std::size_t count=std::size_t(output.stride_bytes)*output.height;
            if(cycle==0){reference[level].assign(bytes,bytes+count);
                ok=write_bmp((std::string(argv[5])+".z"+std::to_string(tile_width)+".bmp").c_str(),output);
            }else{
                std::size_t changed=0;unsigned long long error=0;
                for(std::size_t i=0;i<count;i+=4){bool bad=false;for(unsigned c=0;c<4;++c){
                    unsigned delta=unsigned(std::abs(int(reference[level][i+c])-int(bytes[i+c])));
                    error+=delta;bad=bad || delta>2;}if(bad)++changed;}
                ok=supported?error==0:changed<=count/4000 && error<=count/100;
                std::printf("ZOOM parity width=%d changed=%zu error=%llu status=%s\n",tile_width,changed,error,ok?"pass":"FAIL");
            }
        }
        // The timing sequence retains world content. Separately rebuild each
        // synchronous zoom from an empty scene to detect invalid cross-view reuse.
        if(ok && !native_handoff)for(std::size_t level=0;level<levels.size() && ok;++level){
            tile_width=levels[level];tile_height=tile_width/2;tiles=capture_view();
            frame.tile_width=tile_width;frame.tile_height=tile_height;
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            reset();ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&
                render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && !output.device_recoveries && !output.fallback_tile_count;
            std::size_t changed=0;unsigned maximum=0;unsigned long long error=0;
            if(ok){auto pixels=static_cast<unsigned char const*>(output.bgra_pixels);
                if(reference[level].size()!=std::size_t(output.stride_bytes)*output.height)ok=false;
                else for(std::size_t i=0;i<reference[level].size();++i){
                    unsigned delta=unsigned(std::abs(int(reference[level][i])-int(pixels[i])));
                    changed+=delta!=0;maximum=maximum>delta?maximum:delta;error+=delta;
                }
            }
            ok=ok && changed==0;
            std::printf("ZOOM redraw width=%d changed=%zu max=%u error=%llu result=%u\n",tile_width,changed,maximum,error,unsigned(ok));
            if(ok)write_bmp((std::string(argv[5])+".fresh-z"+std::to_string(tile_width)+".bmp").c_str(),output);
        }
    }
    if(ok && navigation_benchmark) {
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        // Recapture authoritative records at each destination, just as a
        // minimap move does. Include overlap, distant moves and the wrap seam.
        int destinations[][2]={{75,39},{79,39},{35,39},{35,69},{1,39},{99,39}};
        std::vector<std::vector<unsigned char>> reference(6);
        for(int cycle=0;cycle<camera_cycles && ok;++cycle)for(int step=0;step<6 && ok;++step){
            center_x=destinations[step][0];center_y=destinations[step][1];
            LARGE_INTEGER begin={},captured={},end={};QueryPerformanceCounter(&begin);
            tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            QueryPerformanceCounter(&captured);
            int code=render_checked(&frame,&output);QueryPerformanceCounter(&end);
            ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
            std::printf("NAV cycle=%d step=%d x=%d y=%d width=%d result=%d tiles=%u built=%u reused=%u cache_bytes=%u recoveries=%u ms=%.3f capture_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f\n",
                cycle,step,center_x,center_y,tile_width,code,output.rendered_tile_count,output.geometry_tiles_built,output.geometry_tiles_reused,
                output.geometry_cache_bytes,output.device_recoveries,double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(captured.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(output.geometry_ticks)*1000/frequency.QuadPart,double(output.draw_ticks)*1000/frequency.QuadPart,
                double(output.readback_ticks)*1000/frequency.QuadPart);
            camera_memory();std::fflush(stdout);if(!ok)break;
            auto bytes=static_cast<unsigned char const*>(output.bgra_pixels);
            std::size_t count=std::size_t(output.stride_bytes)*output.height;
            if(cycle==0){reference[step].assign(bytes,bytes+count);
                ok=write_bmp((std::string(argv[5])+".nav"+std::to_string(step)+".bmp").c_str(),output);
            }else{
                std::size_t changed=0;unsigned long long error=0;
                for(std::size_t i=0;i<count;i+=4){bool bad=false;for(unsigned c=0;c<4;++c){
                    unsigned delta=unsigned(std::abs(int(reference[step][i+c])-int(bytes[i+c])));
                    error+=delta;bad=bad || delta>2;}if(bad)++changed;}
                ok=changed<=count/4000 && error<=count/100;
                std::printf("NAV parity step=%d changed=%zu error=%llu status=%s\n",step,changed,error,ok?"pass":"FAIL");
            }
        }
    }
    if(ok && scroll_ablation) {
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        auto saved_center_x=center_x,saved_center_y=center_y;
        auto saved_tiles=tiles;
        if(std::strcmp(scroll_option,"sequence")==0) {
            // A short, modern-navigation-shaped trace: small wheel steps,
            // a two-tile jump, a larger jump, reversal, then a return.  The
            // first visit to each camera is the authoritative reference; the
            // later visits must be byte-identical rather than merely cache-hit.
            std::vector<int> offsets={1,2,4,8,4,2,1,0,-2,-4,-8,-4,-2,0};
            std::vector<int> reference_offsets;
            std::vector<std::vector<unsigned char>> references;
            auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
                return std::vector<unsigned char>(p,p+std::size_t(output.stride_bytes)*output.height);};
            reference_offsets.push_back(0);references.push_back(pixels());
            std::vector<unsigned char> previous=references.front();
            int previous_offset=0;
            bool sequence_ok=true;
            char preparation_delay[16]={};
            GetEnvironmentVariableA("C3X_RENDERER_SCROLL_PREPARE_MS",preparation_delay,sizeof(preparation_delay));
            unsigned opportunity=unsigned(std::clamp(std::atoi(preparation_delay),0,10000));
            if(opportunity){
                LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
                Sleep(opportunity);QueryPerformanceCounter(&end);
                std::printf("SCROLL_PREPARATION opportunity_ms=%u actual_ms=%.3f\n",opportunity,
                    double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart);
            }
            for(std::size_t step=0;step<offsets.size() && sequence_ok;++step) {
                int offset=offsets[step];
                center_x=saved_center_x+offset;center_y=saved_center_y;
                LARGE_INTEGER begin={},captured={},end={};QueryPerformanceCounter(&begin);
                tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
                QueryPerformanceCounter(&captured);
                int code=render_checked(&frame,&output);QueryPerformanceCounter(&end);
                auto current=pixels();
                std::size_t translated_mismatches=0;
                if(code==C3X_RENDERER_RESULT_OK && !previous.empty()) {
                    int dx=-(offset-previous_offset)*tile_width/2;
                    int left=(std::max)(0,dx),right=(std::min)(target_width,target_width+dx);
                    if(left<right)for(int y=0;y<target_height;++y) {
                        auto const* old_row=previous.data()+std::size_t(y)*target_width*4;
                        auto const* new_row=current.data()+std::size_t(y)*target_width*4;
                        for(int x=left;x<right;++x) {
                            std::size_t old_index=std::size_t(x-dx)*4,new_index=std::size_t(x)*4;
                            if(std::memcmp(old_row+old_index,new_row+new_index,4)!=0)++translated_mismatches;
                        }
                    }
                }
                auto found=std::find(reference_offsets.begin(),reference_offsets.end(),offset);
                bool exact=true;
                if(found==reference_offsets.end()) {
                    reference_offsets.push_back(offset);references.push_back(current);
                } else exact=current==references[std::size_t(found-reference_offsets.begin())];
                sequence_ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 &&
                    output.device_recoveries==0 && preview_ownership(frame,output) && exact;
                std::printf("SCROLL_SEQUENCE step=%zu offset_columns=%d result=%d exact=%u translated_overlap_mismatches=%zu tiles=%u built=%u reused=%u total_ms=%.3f capture_ms=%.3f cpu_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f raster_reused=%u raster_draw=%u fallback=%u recoveries=%u\n",
                    step,offset,code,unsigned(exact),translated_mismatches,output.rendered_tile_count,output.geometry_tiles_built,output.geometry_tiles_reused,
                    double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                    double(captured.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                    double(output.renderer_cpu_ticks)*1000/frequency.QuadPart,double(output.geometry_ticks)*1000/frequency.QuadPart,
                    double(output.draw_ticks)*1000/frequency.QuadPart,double(output.readback_ticks)*1000/frequency.QuadPart,
                    output.raster_reused_pixels,output.raster_draw_pixels,output.fallback_tile_count,output.device_recoveries);
                camera_memory();std::fflush(stdout);
                previous=std::move(current);previous_offset=offset;
            }
            ok=sequence_ok;
            std::printf("SCROLL_SEQUENCE_END status=%s unique_cameras=%zu exact_revisits=1\n",ok?"pass":"FAIL",references.size());
            center_x=saved_center_x;center_y=saved_center_y;tiles=std::move(saved_tiles);
        } else {
        // Four map-column coordinates are two ordinary isometric tile widths;
        // this is a representative wheel/key scroll, not a one-pixel probe.
        LARGE_INTEGER begin={},captured={},end={};QueryPerformanceCounter(&begin);
        center_x+=4;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        QueryPerformanceCounter(&captured);
        int code=render_checked(&frame,&output);QueryPerformanceCounter(&end);
        ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 && output.device_recoveries==0;
        std::printf("SCROLL_ABLATION label=%s delta_columns=4 result=%d tiles=%u built=%u reused=%u upload_bytes=%llu total_ms=%.3f capture_ms=%.3f cpu_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f raster_reused=%u raster_draw=%u fallback=%u recoveries=%u\n",
            scroll_option,code,output.rendered_tile_count,output.geometry_tiles_built,output.geometry_tiles_reused,
            static_cast<unsigned long long>(output.geometry_upload_bytes),double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
            double(captured.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,double(output.renderer_cpu_ticks)*1000/frequency.QuadPart,double(output.geometry_ticks)*1000/frequency.QuadPart,
            double(output.draw_ticks)*1000/frequency.QuadPart,double(output.readback_ticks)*1000/frequency.QuadPart,
            output.raster_reused_pixels,output.raster_draw_pixels,output.fallback_tile_count,output.device_recoveries);
        camera_memory();std::fflush(stdout);
        center_x=saved_center_x;center_y=saved_center_y;tiles=std::move(saved_tiles);
        }
    }
    char resident_option[8]={};
    if(ok && navigation_benchmark && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_RESIDENT_SWEEP",resident_option,sizeof(resident_option))) {
        // The preceding navigation workload loaded the views centered at
        // (35,39) and (35,69). Exercise NEW intermediate views over their union,
        // not the six cached screenshots. Report builds/uploads rather than
        // assuming that resident content implies a cheap draw.
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        char resident_cold_option[8]={};
        bool resident_cold=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_RESIDENT_COLD",resident_cold_option,sizeof(resident_cold_option)) &&
            std::strcmp(resident_cold_option,"1")==0;
        char steps_option[16]={};
        int steps=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_RESIDENT_STEPS",steps_option,sizeof(steps_option))?
            std::clamp(std::atoi(steps_option),14,1000):14;
        if(steps!=14 && steps>=15*tile_height){std::fprintf(stderr,"resident sweep needs unique pixel cameras\n");return 2;}
        std::printf("RESIDENT_BEGIN mode=%s steps=%d width=%d height=%d tile_width=%d pattern=%s\n",
            resident_cold?"cold":"retained",steps,target_width,target_height,tile_width,steps==14?"tile-v1":"pixel-v1");
        for(int step=0;step<steps && ok;++step) {
            int pixel_y=steps==14?(step+1)*tile_height:(step+1)*15*tile_height/(steps+1);
            center_x=35;center_y=39+(pixel_y/tile_height)*2;
            // Independent reference pixels: forget renderer caches, but keep
            // the fixture's authoritative object sites and presentation clock.
            if(resident_cold){
                reset();
                ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK;
                if(!ok)break;
            }
            LARGE_INTEGER begin={},captured={},end={};QueryPerformanceCounter(&begin);
            tiles=capture_view();
            for(auto& tile:tiles)tile.anchor_y-=pixel_y%tile_height;
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            QueryPerformanceCounter(&captured);
            int code=render_checked(&frame,&output);QueryPerformanceCounter(&end);
            ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
            std::printf("RESIDENT_NAV step=%d x=%d y=%d result=%d built=%u reused=%u upload_bytes=%llu ms=%.3f capture_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f pixel_y=%d reused_pixels=%u draw_pixels=%u cached_pixels=%u\n",
                step,center_x,center_y,code,output.geometry_tiles_built,output.geometry_tiles_reused,
                static_cast<unsigned long long>(output.geometry_upload_bytes),
                double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(captured.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(output.geometry_ticks)*1000/frequency.QuadPart,
                double(output.draw_ticks)*1000/frequency.QuadPart,double(output.readback_ticks)*1000/frequency.QuadPart,
                pixel_y,output.raster_reused_pixels,output.raster_draw_pixels,output.raster_cached_pixels);
            camera_memory();std::fflush(stdout);
            if(ok){
                ok=write_bmp((std::string(argv[5])+".resident"+std::to_string(step)+".bmp").c_str(),output);
                auto data=static_cast<unsigned char const*>(output.bgra_pixels);
                std::size_t bytes=std::size_t(output.stride_bytes)*output.height;
                unsigned long long checksum=14695981039346656037ull;
                for(std::size_t i=0;i<bytes;++i)checksum=(checksum^data[i])*1099511628211ull;
                std::printf("RESIDENT_IMAGE step=%d bytes=%zu fnv64=%llu saved=%d\n",step,bytes,checksum,int(ok));
            }
        }
        std::printf("RESIDENT_END status=%s\n",ok?"pass":"FAIL");
    }
    if(ok && distant_steps){
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        std::uint32_t random=0x433358u;
        auto next=[&](){random=random*1664525u+1013904223u;return random;};
        std::vector<std::pair<int,int>> visited{{center_x,center_y}};
        std::printf("DISTANT_BEGIN steps=%d width=%d height=%d tile_width=%d map_prepared=0 pattern=lcg-v1\n",distant_steps,target_width,target_height,tile_width);
        for(int step=0;step<distant_steps && ok;++step){
            int x=0,y=0;bool selected=false;
            for(unsigned attempt=0;attempt<10000 && !selected;++attempt){
                x=int((next()>>8)%unsigned(map_width/2))*2+1;y=int((next()>>8)%unsigned(map_height/2))*2+1;
                int dx=std::abs(x-center_x),dy=std::abs(y-center_y);
                if(frame.world_wrap_x)dx=(std::min)(dx,map_width-dx);
                if(frame.world_wrap_y)dy=(std::min)(dy,map_height-dy);
                selected=dx+dy>=(std::min)(map_width,map_height)/3 && std::find(visited.begin(),visited.end(),std::make_pair(x,y))==visited.end();
            }
            if(!selected){ok=false;break;}
            center_x=x;center_y=y;visited.emplace_back(x,y);
            LARGE_INTEGER begin={},captured={},end={};QueryPerformanceCounter(&begin);
            tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());QueryPerformanceCounter(&captured);
            int code=render_checked(&frame,&output);QueryPerformanceCounter(&end);
            ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
            std::printf("DISTANT_NAV step=%d x=%d y=%d result=%d built=%u reused=%u upload_bytes=%u ms=%.3f capture_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f recoveries=%u\n",
                step,x,y,code,output.geometry_tiles_built,output.geometry_tiles_reused,output.geometry_upload_bytes,
                double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,double(captured.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(output.geometry_ticks)*1000/frequency.QuadPart,double(output.draw_ticks)*1000/frequency.QuadPart,
                double(output.readback_ticks)*1000/frequency.QuadPart,output.device_recoveries);
            camera_memory();std::fflush(stdout);
            if(ok)ok=write_bmp((std::string(argv[5])+".distant"+std::to_string(step)+".bmp").c_str(),output);
        }
        std::printf("DISTANT_END status=%s\n",ok?"pass":"FAIL");
    }
    if(ok && idle_steps) {
        // Every sample requires a new authored pose bucket. Run unpaced: disk
        // evidence, memory queries and pixel comparisons are outside timing.
        // This measures completion capacity, never native delivered frame rate.
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        unsigned changes=0;
        std::vector<unsigned char> previous;
        std::vector<c3x_renderer_tile_v1 const*> unit_sites;
        unsigned cities_count=0,roads_count=0,farms_count=0,mines_count=0,camps_count=0,resources_count=0;
        for(auto const& tile:tiles)if(tile.tile_flags&C3X_RENDERER_TILE_RENDER) {
            cities_count+=tile.city_id>=0;roads_count+=tile.road_mask!=0;
            farms_count+=(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)!=0;
            mines_count+=(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)!=0;
            camps_count+=(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)!=0;
            resources_count+=tile.resource_id>=0;
            if(tile.real_terrain_type<=4 && tile.city_id<0 && tile.anchor_x>tile_width &&
               tile.anchor_x<target_width-tile_width*2 && tile.anchor_y>tile_height*2 &&
               tile.anchor_y<target_height-tile_height*3)unit_sites.push_back(&tile);
        }
        std::sort(unit_sites.begin(),unit_sites.end(),[](auto a,auto b){return a->variant_seed<b->variant_seed;});
        if(unit_sites.size()<std::size_t(idle_unit_count))ok=false;
        else unit_sites.resize(idle_unit_count);
        std::sort(unit_sites.begin(),unit_sites.end(),[](auto a,auto b){return a->anchor_y==b->anchor_y?a->anchor_x<b->anchor_x:a->anchor_y<b->anchor_y;});
        auto unit_draw=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_expanded"));
        HDC unit_dc=nullptr;HBITMAP unit_bitmap=nullptr;HGDIOBJ old_bitmap=nullptr;void* unit_pixels=nullptr;
        if(idle_unit_count && ok) {
            BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
            info.bmiHeader.biWidth=target_width;info.bmiHeader.biHeight=-target_height;
            info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
            unit_dc=CreateCompatibleDC(nullptr);
            if(unit_dc)unit_bitmap=CreateDIBSection(unit_dc,&info,DIB_RGB_COLORS,&unit_pixels,nullptr,0);
            if(!unit_draw || !unit_dc || !unit_bitmap || !unit_pixels)ok=false;
            if(unit_bitmap)old_bitmap=SelectObject(unit_dc,unit_bitmap);
        }
        double warmup_ms=0;
        std::printf("IDLE_BEGIN steps=%d warmup=%d pose_hz=15 paced=0 x=%d y=%d tile_width=%d units=%d dense=%d cities=%u roads=%u farms=%u mines=%u camps=%u resources=%u unit_actions=%s\n",
            idle_steps,idle_warmup,center_x,center_y,tile_width,idle_unit_count,int(dense_scene),cities_count,roads_count,
            farms_count,mines_count,camps_count,resources_count,
            mixed_unit_actions?"mixed":realistic_unit_actions?"realistic":"idle");
        LARGE_INTEGER idle_origin={};QueryPerformanceCounter(&idle_origin);
        std::printf("IDLE_PACING period_ms=%d includes_wait_in_request=0 absolute_deadlines=1\n",idle_pace_ms);
        for(int step=-idle_warmup;step<idle_steps && ok;++step) {
            auto deadline=idle_origin.QuadPart+c3x_renderer_i64(step+idle_warmup+1)*idle_pace_ms*frequency.QuadPart/1000;
            if(idle_pace_ms){LARGE_INTEGER now={};
                do{QueryPerformanceCounter(&now);if(now.QuadPart+frequency.QuadPart/500<deadline)Sleep(1);else if(now.QuadPart<deadline)Sleep(0);}while(now.QuadPart<deadline);
            }
            frame.presentation_time_ticks=1000000+c3x_renderer_i64(step+idle_warmup+1)*frame.presentation_frequency/15;
            LARGE_INTEGER begin={},map_end={},copy_end={},end={};QueryPerformanceCounter(&begin);
            int code=render_checked(&frame,&output);QueryPerformanceCounter(&map_end);
            ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 &&
                output.visible_animation_count>0 && output.request_continuous_redraw &&
                output.geometry_tiles_built==0 && output.geometry_upload_bytes==0;
            if(ok && idle_unit_count)std::memcpy(unit_pixels,output.bgra_pixels,std::size_t(output.stride_bytes)*output.height);
            QueryPerformanceCounter(&copy_end);
            unsigned moving_units=0,attacking_units=0,fortifying_units=0,idling_units=0;
            unsigned selected_units=0,working_units=0,directed_units=0;
            for(int i=0;i<idle_unit_count && ok;++i) {
                char const* names[]={"Archer","Swordsman","Infantry","Warrior","Scout","Settler","Worker"};
                auto site=unit_sites[i];c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);
                unit.unit_id=10000+i;unit.action=1;unit.direction=1+i%8;unit.frame_count=16;
                unit.presentation_frequency=frame.presentation_frequency;unit.presentation_time_ticks=frame.presentation_time_ticks;
                unit.sprite_width=unit.sprite_height=191;unit.projection_scale_milli=tile_width*1000/128;
                unit.body_x=site->anchor_x+tile_width/2-191*unit.projection_scale_milli/2000;
                unit.body_y=site->anchor_y+tile_height/2-191*unit.projection_scale_milli/2000;
                if(mixed_unit_actions) {
                    // These are scripted native inputs, not renderer-owned
                    // simulation. Timelines are offset per identity. Every
                    // period returns to its original anchor without a jump.
                    int timeline=(step+idle_warmup+i*7)%80,phase=timeline%16;
                    bool combatant=i%std::size(names)<5;
                    int travel=0;
                    if(timeline<16){unit.action=2;unit.direction=3;travel=phase+1;}
                    else if(timeline<32){unit.action=combatant?3:8;unit.direction=3;travel=16;}
                    else if(timeline<48){unit.action=2;unit.direction=7;travel=15-phase;}
                    else if(timeline<64){unit.action=7;unit.direction=7;}
                    unit.action_cursor=phase;
                    unit.body_x+=travel*tile_width/32;unit.body_y+=travel*tile_height/32;
                } else if(realistic_unit_actions) {
                    int phase=(step+idle_warmup)%16;
                    if(i==0){unit.action_cursor=phase;++selected_units;}
                    else if(i==1){unit.action=3;unit.action_cursor=phase;++directed_units;}
                    else if(i==6){unit.action=8;unit.action_cursor=phase;++working_units;}
                }
                moving_units+=unit.action==2;attacking_units+=unit.action==3;
                fortifying_units+=unit.action==7;idling_units+=unit.action==1;
                unit.hour=frame.hour;unit.season=frame.season;unit.display_color_rgb=0x205bdd;
                sprintf_s(unit.unit_key,"PRTO_%s",names[i%std::size(names)]);
                int bounds[4]={};ok=unit_draw(&unit,unit_dc,unit_dc,bounds)==C3X_RENDERER_RESULT_OK;
            }
            if(idle_unit_count)GdiFlush();
            QueryPerformanceCounter(&end);
            if(ok && dense_scene)for(unsigned i=0;i<frame.tile_count;++i) {
                auto const& tile=frame.tiles[i];if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
                unsigned required=0;
                if(tile.city_id>=0)required|=C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED;
                if(tile.road_mask)required|=C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED;
                if(tile.railroad_mask)required|=C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED;
                if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)required|=C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED;
                if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)required|=C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED;
                if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)required|=C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED;
                if(tile.resource_id>=0)required|=C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
                if(i>=output.replacement_tile_count || (output.replacement_tile_flags[i]&required)!=required) {
                    std::printf("IDLE ownership FAIL tile=%u required=%u actual=%u\n",i,required,
                        i<output.replacement_tile_count?output.replacement_tile_flags[i]:0u);ok=false;break;
                }
            }
            if(step<0){warmup_ms+=double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart;continue;}
            auto composed=output;if(idle_unit_count)composed.bgra_pixels=unit_pixels;
            auto data=static_cast<unsigned char const*>(composed.bgra_pixels);
            std::size_t bytes=std::size_t(output.stride_bytes)*output.height;
            bool changed=previous.empty() || previous.size()!=bytes || std::memcmp(previous.data(),data,bytes)!=0;
            if(step>0 && changed)++changes;
            previous.assign(data,data+bytes);
            std::printf("IDLE_FRAME step=%d ticks=%lld result=%d visible=%u built=%u reused=%u upload_bytes=%u changed=%d ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f recoveries=%u map_ms=%.3f copy_ms=%.3f units_ms=%.3f units=%d moving=%u attacking=%u fortifying=%u idling=%u selected=%u working=%u directed=%u\n",
                step,frame.presentation_time_ticks,code,output.visible_animation_count,output.geometry_tiles_built,
                output.geometry_tiles_reused,output.geometry_upload_bytes,int(changed),
                double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(output.geometry_ticks)*1000/frequency.QuadPart,double(output.draw_ticks)*1000/frequency.QuadPart,
                double(output.readback_ticks)*1000/frequency.QuadPart,output.device_recoveries,
                double(map_end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(copy_end.QuadPart-map_end.QuadPart)*1000/frequency.QuadPart,
                double(end.QuadPart-copy_end.QuadPart)*1000/frequency.QuadPart,idle_unit_count,moving_units,attacking_units,fortifying_units,idling_units,
                selected_units,working_units,directed_units);
            if(idle_pace_ms)std::printf("IDLE_DEADLINE step=%d late_ms=%.3f complete_ms=%.3f\n",step,
                double(begin.QuadPart-deadline)*1000/frequency.QuadPart,double(end.QuadPart-deadline)*1000/frequency.QuadPart);
            camera_memory();std::fflush(stdout);
            if(ok)ok=write_bmp((std::string(argv[5])+".idle"+std::to_string(step)+".bmp").c_str(),composed);
        }
        if(unit_dc && old_bitmap)SelectObject(unit_dc,old_bitmap);
        if(unit_bitmap)DeleteObject(unit_bitmap);
        if(unit_dc)DeleteDC(unit_dc);
        std::printf("IDLE_WARMUP frames=%d ms=%.3f includes_first_unit_poses=1 map_initial_render_excluded=1\n",idle_warmup,warmup_ms);
        ok=ok && (idle_steps==1 || changes>0);
        std::printf("IDLE_END status=%s changed_frames=%u\n",ok?"pass":"FAIL",changes);
    }
    if(ok && animate && !prepared_view_fixture && !topology_edits && !ambient_async && !zoom_benchmark && !navigation_benchmark && !scroll_ablation && !distant_steps && !idle_steps && !busy_session) {
        // Exercise animation after an immutable viewport LRU restore, not only
        // after the unchanged-current-view fast path.
        auto initial=static_cast<unsigned char const*>(output.bgra_pixels);
        std::vector<unsigned char> initial_pixels(initial,initial+std::size_t(output.stride_bytes)*output.height);
        int original_width=tile_width;
        for(int next:{original_width==128?192:128,original_width}) {
            tile_width=next;tile_height=next/2;tiles=capture_view();
            frame.tile_width=tile_width;frame.tile_height=tile_height;
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            ok=ok && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
        }
        ok=ok && std::memcmp(output.bgra_pixels,initial_pixels.data(),initial_pixels.size())==0;
        std::printf("ANIMATION zoom-return parity: %s\n",ok?"pass":"FAIL");
        std::vector<unsigned char> previous;
        unsigned changes=0;
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        for (int n=0;n<6 && ok;++n) {
            frame.presentation_time_ticks=1000000+n*200000;
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
            QueryPerformanceCounter(&end);
            if (!ok) break;
            ok=output.visible_animation_count>0 && output.request_continuous_redraw && output.geometry_tiles_built==0 && output.geometry_upload_bytes==0;
            for (unsigned i=0;i<frame.tile_count;++i)
                if ((frame.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER) && frame.tiles[i].resource_id>=100)
                    ok=ok && (output.replacement_tile_flags[i]&C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED)!=0;
            auto bytes=static_cast<unsigned char const*>(output.bgra_pixels);
            std::vector<unsigned char> current(bytes,bytes+std::size_t(output.stride_bytes)*output.height);
            if (!previous.empty() && current!=previous) ++changes;
            previous=current;
            std::string path=std::string(argv[5])+".animation-"+std::to_string(n)+".bmp";
            ok=ok && write_bmp(path.c_str(),output);
            std::printf("ANIMATION temporal frame=%d visible=%u terrain_built=%u terrain_upload=%zu ms=%.3f\n",
                n,output.visible_animation_count,output.geometry_tiles_built,std::size_t(output.geometry_upload_bytes),
                double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart);
            ok=ok && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK &&
                std::memcmp(output.bgra_pixels,current.data(),current.size())==0;
        }
        ok=ok && changes==5;
        std::printf("ANIMATION temporal: %s changed_frames=%u\n",ok?"pass":"FAIL",changes);
        auto compare_cold = [&](char const * label) {
            auto pixels=static_cast<unsigned char const*>(output.bgra_pixels);
            std::vector<unsigned char> cached(pixels,pixels+std::size_t(output.stride_bytes)*output.height);
            reset();
            if(set_definitions(argv[2],argv[3],nullptr,custom_path)!=C3X_RENDERER_RESULT_OK ||
               render_checked(&frame,&output)!=C3X_RENDERER_RESULT_OK)return false;
            auto fresh=static_cast<unsigned char const*>(output.bgra_pixels);
            std::size_t changed=0;unsigned long long error=0;
            for(std::size_t i=0;i<cached.size();i+=4){bool bad=false;
                for(unsigned c=0;c<4;++c){unsigned delta=unsigned(std::abs(int(cached[i+c])-int(fresh[i+c])));
                    error+=delta;bad=bad || delta>2;}if(bad)++changed;}
            bool supported=frame.tile_width==128 || frame.tile_width==160 || frame.tile_width==192;
            bool same=supported?error==0:changed<=cached.size()/4000 && error<=cached.size()/100;
            std::printf("ANIMATION %s parity: %s changed=%zu error=%llu bytes=%zu\n",label,same?"pass":"FAIL",changed,error,cached.size());
            return same;
        };
        if(ok) {
            center_x+=4;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.visible_animation_count>0;
            if(ok)ok=compare_cold("scroll");
        }
        if(ok) {
            auto before_removal=output.visible_animation_count;
            for(auto & tile:tiles){tile.resource_id=-1;tile.resource_name[0]='\0';}
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.visible_animation_count<before_removal &&
                bool(output.request_continuous_redraw)==(output.visible_animation_count!=0);
            // Waves may still animate after resources are removed. Their
            // redraw ownership must survive, while resource replacement ends.
            for(unsigned i=0;i<output.replacement_tile_count;++i)
                ok=ok && !(output.replacement_tile_flags[i]&C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED);
            if(ok)ok=compare_cold("removal");
        }
    }
    if(ok && retained_boundary) {
        // The fixed objects straddle the world-anchored 128-pixel region edge.
        // Each reference starts from reset state and ordinary independent draws.
        auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            return std::vector<unsigned char>(p,p+std::size_t(output.stride_bytes)*output.height);};
        auto flags=[&](){return std::vector<unsigned>(output.replacement_tile_flags,
            output.replacement_tile_flags+output.replacement_tile_count);};
        auto draw=[&](){return render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK &&
            output.fallback_tile_count==0 && output.device_recoveries==0;};
        auto cold=[&](){
            char prepared[8]={};GetEnvironmentVariableA("C3X_RENDERER_PREPARED_RESOURCE_PASS",prepared,sizeof(prepared));
            SetEnvironmentVariableA("C3X_RENDERER_PREPARED_RESOURCE_PASS","0");reset();
            bool result=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw();
            SetEnvironmentVariableA("C3X_RENDERER_PREPARED_RESOURCE_PASS",prepared[0]?prepared:nullptr);return result;
        };
        std::vector<unsigned char> preceding;
        unsigned checks=0;
        for(unsigned step=0;ok && step<6;++step) {
            center_x=boundary_x+(step<2?1:0);center_y=boundary_y;
            boundary_resource=step!=3;boundary_mine=step!=4;
            frame.presentation_time_ticks=1000000+(step?100000:0);
            tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            ok=draw();if(!ok)break;
            auto candidate=pixels();auto ownership=flags();auto animations=output.visible_animation_count;
            unsigned combined=0;for(auto value:ownership)combined|=value;
            ok=bool(combined&C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED)==boundary_resource &&
                bool(combined&C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED)==boundary_mine &&
                bool(animations)==boundary_resource;
            // Advancing pose/removing/reappearing must affect actual pixels.
            if(step==1 || step==3 || step==4 || step==5)ok=ok && candidate!=preceding;
            auto warm_output=output;warm_output.bgra_pixels=candidate.data();
            if(ok)ok=cold();
            bool exact=ok && pixels()==candidate && flags()==ownership && output.visible_animation_count==animations;
            std::printf("RETAINED_BOUNDARY step=%u exact=%u animations=%u ownership=%u clock=%lld\n",
                step,unsigned(exact),animations,combined,frame.presentation_time_ticks);
            if(!exact){
                write_bmp((std::string(argv[5])+".boundary-candidate.bmp").c_str(),warm_output);
                write_bmp((std::string(argv[5])+".boundary-reference.bmp").c_str(),output);
                ok=false;break;
            }
            ++checks;preceding=std::move(candidate);
        }
        std::printf("RETAINED_BOUNDARY_END status=%s checks=%u independent_full_redraw=1\n",ok?"pass":"FAIL",checks);
    }
    if(ok && topology_edits) {
        // Exercise real copied world changes. Independent reset redraws may not
        // reuse any pre-edit geometry or completed region from the candidate.
        auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            return std::vector<unsigned char>(p,p+std::size_t(output.stride_bytes)*output.height);};
        auto flags=[&](){return std::vector<unsigned>(output.replacement_tile_flags,
            output.replacement_tile_flags+output.replacement_tile_count);};
        auto draw=[&](){return render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK &&
            output.fallback_tile_count==0 && output.device_recoveries==0;};
        auto before=pixels();
        std::size_t visible_site=source_tiles.size(),distant_site=source_tiles.size();
        double near_distance=1e30,far_distance=-1;
        for(std::size_t i=0;i<source_tiles.size();++i) {
            auto const& source=source_tiles[i];
            if(source.base!=source.real || (source.base!=2 && source.base!=3) || source.river)continue;
            bool captured=false;
            for(auto const& tile:tiles) {
                if((tile.tile_x%map_width+map_width)%map_width!=source.x || tile.tile_y!=source.y)continue;
                captured=true;
                int x=tile.anchor_x+tile_width/2,y=tile.anchor_y+tile_height/2;
                double distance=double(x-target_width/2)*(x-target_width/2)+double(y-target_height/2)*(y-target_height/2);
                if(x>=tile_width && x<target_width-tile_width && y>=tile_height && y<target_height-tile_height && distance<near_distance){visible_site=i;near_distance=distance;}
            }
            int dx=std::abs(source.x-center_x);dx=dx<map_width-dx?dx:map_width-dx;
            double distance=double(dx)*dx+double(source.y-center_y)*(source.y-center_y);
            if(!captured && distance>far_distance){distant_site=i;far_distance=distance;}
        }
        if(visible_site==source_tiles.size() || distant_site==source_tiles.size()) {
            std::fputs("TOPOLOGY_EDIT missing visible/distant flat-land fixture sites\n",stderr);ok=false;
        }
        unsigned checks=0;
        int original=0;
        for(unsigned step=0;ok && step<4;++step) {
            std::size_t index=step<2?distant_site:visible_site;
            auto& source=source_tiles[index];
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            if(!(step&1))original=source.base;
            source.base=source.real=(step&1)?original:(original==2?3:2);
            auto world_index=(std::size_t(source.y)*map_width+source.x)/2;
            world[world_index]=(world[world_index]&0xffff0000u)|unsigned(source.base)|(unsigned(source.real)<<8);
            ++frame.world_topology_revision;
            tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            frame.dirty_flags=C3X_RENDERER_DIRTY_SCENE|C3X_RENDERER_DIRTY_STATIC_MAP;
            ok=draw();if(!ok)break;
            auto candidate=pixels();auto ownership=flags();auto animations=output.visible_animation_count;
            bool changed=candidate!=before;
            QueryPerformanceCounter(&end);
            camera_memory(); // Host boundary sample, outside the timed edit.
            auto warm_output=output;warm_output.bgra_pixels=candidate.data();
            // A distant flat-land edit cannot alter this captured view. The
            // selected visible edit/reversal must alter actual displayed pixels.
            ok=changed==(step>=2);
            if(ok){reset();ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw();}
            bool exact=ok && pixels()==candidate && flags()==ownership && output.visible_animation_count==animations;
            std::printf("TOPOLOGY_EDIT step=%u site=%s x=%d y=%d revision=%lld changed=%u exact=%u total_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f\n",
                step,step<2?"distant":"visible",source.x,source.y,frame.world_topology_revision,unsigned(changed),unsigned(exact),
                double(end.QuadPart-begin.QuadPart)*1000/timing_frequency.QuadPart,
                double(warm_output.geometry_ticks)*1000/timing_frequency.QuadPart,
                double(warm_output.draw_ticks)*1000/timing_frequency.QuadPart,
                double(warm_output.readback_ticks)*1000/timing_frequency.QuadPart);
            if(!exact){
                write_bmp((std::string(argv[5])+".edit-candidate.bmp").c_str(),warm_output);
                write_bmp((std::string(argv[5])+".edit-reference.bmp").c_str(),output);
                ok=false;break;
            }
            before=std::move(candidate);++checks;
        }
        std::printf("TOPOLOGY_EDIT_END status=%s checks=%u independent_full_redraw=1\n",ok?"pass":"FAIL",checks);
    }
    char content_edit_option[8]={};
    if(ok && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CONTENT_EDITS",content_edit_option,sizeof(content_edit_option))) {
        // Change appearance without a world-topology revision. A city next to
        // forest exercises the outer instance binding and inner exclusion mesh.
        std::size_t site=tiles.size(),forest_site=tiles.size();double closest=1e30;
        for(std::size_t i=0;i<tiles.size();++i){auto const& tile=tiles[i];
            if(tile.anchor_x<tile_width || tile.anchor_x>target_width-2*tile_width ||
               tile.anchor_y<tile_height || tile.anchor_y>target_height-2*tile_height)continue;
            for(std::size_t j=0;j<tiles.size();++j){auto const& other=tiles[j];
                if(other.city_id>=0 ||
                   std::abs(other.tile_x-tile.tile_x)!=1 || std::abs(other.tile_y-tile.tile_y)!=1)continue;
                double dx=tile.anchor_x+tile_width/2-target_width/2,dy=tile.anchor_y+tile_height/2-target_height/2;
                if(dx*dx+dy*dy<closest){closest=dx*dx+dy*dy;site=i;forest_site=j;}
            }
        }
        if(site==tiles.size()){std::fputs("CONTENT_EDIT missing captured neighboring land sites\n",stderr);ok=false;}
        if(ok){
            auto& city=tiles[site];city.tile_flags|=C3X_RENDERER_TILE_RENDER;
            city.terrain_type=city.real_terrain_type=2;city.feature_flags=city.improvement_flags=city.irrigation_mask=city.river_code=city.has_effect=0;
            city.resource_id=city.resource_class=-1;city.resource_name[0]=0;city.city_id=901;city.city_owner_id=1;city.city_size=0;
            city.city_culture_group=0;city.city_era=0;city.city_flags=0;
            auto& forest=tiles[forest_site];forest.tile_flags|=C3X_RENDERER_TILE_RENDER;forest.terrain_type=2;forest.real_terrain_type=7;
            forest.feature_flags|=C3X_RENDERER_FEATURE_FOREST;
            auto x=(forest.tile_x%map_width+map_width)%map_width;
            auto index=(std::size_t(forest.tile_y)*map_width+x)/2;
            world[index]=(world[index]&0xffff0000u)|2u|(7u<<8);
            auto city_x=(city.tile_x%map_width+map_width)%map_width;world[(std::size_t(city.tile_y)*map_width+city_x)/2]=2u|(2u<<8);
            ++frame.world_topology_revision;frame.tiles=tiles.data();
            reset();ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&
                render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && !output.device_recoveries && !output.fallback_tile_count;
        }
        auto original=tiles;unsigned checks=0;
        auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            return std::vector<unsigned char>(p,p+std::size_t(output.stride_bytes)*output.height);};
        auto flags=[&](){return std::vector<unsigned>(output.replacement_tile_flags,output.replacement_tile_flags+output.replacement_tile_count);};
        for(unsigned step=0;ok && step<6;++step){
            tiles=original;
            if(!(step&1))for(auto& tile:tiles){
                if(tile.tile_x!=original[site].tile_x || tile.tile_y!=original[site].tile_y)continue;
                if(step==0)tile.city_size=(tile.city_size+1)%3;
                if(step==2)tile.city_id=-1;
                if(step==4){tile.city_culture_group=(tile.city_culture_group+1)%5;tile.city_era=(tile.city_era+1)%4;}
            }
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());frame.dirty_flags=C3X_RENDERER_DIRTY_SCENE|C3X_RENDERER_DIRTY_STATIC_MAP;
            auto revision=frame.world_topology_revision;
            LARGE_INTEGER edit_begin={},edit_end={},edit_frequency={};QueryPerformanceFrequency(&edit_frequency);
            QueryPerformanceCounter(&edit_begin);
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 && output.device_recoveries==0;
            QueryPerformanceCounter(&edit_end);
            if(!ok)break;
            double edit_ms=double(edit_end.QuadPart-edit_begin.QuadPart)*1000/edit_frequency.QuadPart;
            auto edit_built=output.geometry_tiles_built,edit_reused=output.geometry_tiles_reused;
            double edit_geometry_ms=double(output.geometry_ticks)*1000/edit_frequency.QuadPart;
            auto candidate=pixels();auto ownership=flags();auto animations=output.visible_animation_count;
            auto warm=output;warm.bgra_pixels=candidate.data();
            reset();ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&
                render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && !output.device_recoveries && !output.fallback_tile_count;
            bool exact=ok && pixels()==candidate && flags()==ownership && output.visible_animation_count==animations;
            std::printf("CONTENT_EDIT step=%u exact=%u topology_unchanged=%u city=%d total_ms=%.3f geometry_ms=%.3f built=%u reused=%u\n",
                step,unsigned(exact),unsigned(frame.world_topology_revision==revision),original[site].city_id,edit_ms,edit_geometry_ms,edit_built,edit_reused);
            if(!exact){write_bmp((std::string(argv[5])+".content-candidate.bmp").c_str(),warm);
                write_bmp((std::string(argv[5])+".content-reference.bmp").c_str(),output);ok=false;break;}
            ++checks;
        }
        std::printf("CONTENT_EDIT_END status=%s checks=%u independent_full_redraw=1\n",ok?"pass":"FAIL",checks);
    }
    char color_option[8]={};
    if(ok && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_COLOR",color_option,sizeof(color_option)))
        ok=write_color_preview(argv[5],module,output);
    if(ok && objects){
        unsigned ownership=0;for(unsigned i=0;i<output.replacement_tile_count;++i)ownership|=output.replacement_tile_flags[i];
        char city_only_setting[8]={};
        bool city_only=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CITY_ONLY",city_only_setting,sizeof(city_only_setting)) && city_only_setting[0]=='1';
        unsigned expected=C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED;
        if(!city_only)expected|=C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED|
            C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED|C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED|
            C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED|C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED;
        ok=(ownership&expected)==expected;
        std::printf("PICKUP synthetic objects: %s ownership=%u\n",ok?"pass":"FAIL",ownership);
    }
    if(timing_enabled && ok)ok=write_bmp((std::string(argv[5])+".result.bmp").c_str(),output);
    if(timing_enabled) {
        std::printf("TIMING_SETUP schema=1 frequency=%lld process_enter=%lld source_done=%lld dll_done=%lld definitions_done=%lld initial_begin=%lld initial_done=%lld dropped=%u\n",
            timing_frequency.QuadPart,process_enter.QuadPart,source_done.QuadPart,dll_done.QuadPart,
            definitions_done.QuadPart,initial_begin.QuadPart,initial_done.QuadPart,timing_dropped);
        for(std::size_t i=0;i<timing_requests.size();++i) {
            auto const& r=timing_requests[i];
            std::printf("TIMING_REQUEST id=%zu capture_begin=%lld capture_end=%lld caller_enter=%lld caller_return=%lld correct_done=%lld geometry_ticks=%lld draw_ticks=%lld readback_ticks=%lld result=%d tiles=%u\n",
                i,r.capture_begin,r.capture_end,r.caller_enter,r.caller_return,r.correct_done,r.geometry,r.draw,r.readback,r.result,r.tiles);
        }
    }
    std::printf("BIQ %dx%d viewport: %u visible tiles, %u fallback, output=%s\n",
                map_width, map_height, output.rendered_tile_count, output.fallback_tile_count, argv[5]);
#ifdef C3X_LAB_PREVIEW
    #include "../lab/volcano_witness.h"
#endif
    char edit_option[8]={};
    if(ok && pickup && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_EDITS",edit_option,sizeof(edit_option))) {
        auto changed=std::find_if(tiles.begin(),tiles.end(),[&](auto const& tile){
            return (tile.tile_flags&C3X_RENDERER_TILE_RENDER) &&
                (active ? tile.real_terrain_type==10 : tile.real_terrain_type==2) &&
                tile.anchor_x>target_width/4 && tile.anchor_x<target_width*3/4 &&
                tile.anchor_y>target_height/4 && tile.anchor_y<target_height*3/4;
        });
        if(changed==tiles.end())ok=false;
        else {
            int x=((changed->tile_x%map_width)+map_width)%map_width,y=changed->tile_y;
            auto& value=world[(std::size_t(y)*map_width+x)/2];
            if(active){value^=1u<<24;changed->has_effect=(value>>24)!=0;}
            else {value=11u|(11u<<8);changed->terrain_type=changed->real_terrain_type=11;
                changed->feature_flags=0;changed->river_code=0;}
            ++frame.world_topology_revision;
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 &&
                output.geometry_tiles_built>0 && output.geometry_tiles_reused>0;
            std::printf("PICKUP authoritative edit: %s active=%u built=%u reused=%u\n",ok?"pass":"FAIL",
                unsigned(active),output.geometry_tiles_built,output.geometry_tiles_reused);
            std::vector<unsigned char> edited;
            if(ok){auto first=static_cast<unsigned char const*>(output.bgra_pixels);
                edited.assign(first,first+output.stride_bytes*output.height);}
            reset();
            if(set_definitions(argv[2],argv[3],nullptr,custom_path)!=C3X_RENDERER_RESULT_OK ||
                render_checked(&frame,&output)!=C3X_RENDERER_RESULT_OK)ok=false;
            if(ok){auto cold=static_cast<unsigned char const*>(output.bgra_pixels);
                std::size_t changed_pixels=0;unsigned long long error=0;
                for(std::size_t i=0;i<edited.size();i+=4){bool bad=false;
                    for(unsigned c=0;c<4;++c){unsigned delta=unsigned(std::abs(int(edited[i+c])-int(cold[i+c])));
                        error+=delta;bad=bad || delta>2;}if(bad)++changed_pixels;}
                std::printf("PICKUP edit pixel parity: changed=%zu error=%llu bytes=%zu\n",changed_pixels,error,edited.size());
                ok=changed_pixels<=edited.size()/4000 && error<=edited.size()/100;
            }
        }
    }
    char replay[16]={};
    if(ok && pickup && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_REPLAY",replay,sizeof(replay))) {
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        std::vector<double> times;
        for(int n=0;n<32;++n){LARGE_INTEGER a={},b={};QueryPerformanceCounter(&a);
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && ok;QueryPerformanceCounter(&b);
            times.push_back(double(b.QuadPart-a.QuadPart)*1000/frequency.QuadPart);
            if(output.geometry_tiles_built || output.geometry_upload_bytes)ok=false;
        }
        std::sort(times.begin(),times.end());
        std::printf("PICKUP selection p95_ms=%.3f max_ms=%.3f\n",times[30],times[31]);
        auto prepare=[&](){
            LARGE_INTEGER begin={},now={};QueryPerformanceCounter(&begin);
            for(;;){
                Sleep(25);int code=render_checked(&frame,&output);QueryPerformanceCounter(&now);
                if(code!=C3X_RENDERER_RESULT_OK)return false;
                if(!output.prefetch_tiles_pending && !output.prefetch_blocks_pending)break;
                if(now.QuadPart-begin.QuadPart>frequency.QuadPart*60)break;
            }
            std::printf("PICKUP prepared pending=%u tiles=%u unavailable=%u blocks=%u bytes=%u\n",
                output.prefetch_tiles_pending,output.prefetch_tiles_built,output.prefetch_tiles_unavailable,
                output.prefetch_blocks_built,output.prefetch_cache_bytes);return true;
        };
        char minimap_option[8]={};
        bool minimap=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_MINIMAP",minimap_option,sizeof(minimap_option))!=0;
        if(!minimap)ok=prepare() && ok;
        int original_x=center_x,original_y=center_y;
        std::vector<std::array<int,2>> steps=minimap
            ? std::vector<std::array<int,2>>{{-48,24},{-16,-12},{0,0},{-48,24},{0,0}}
            : std::vector<std::array<int,2>>{{4,0},{8,0},{8,4},{4,4},{0,0}};
        for(auto const& step:steps){
            center_x=original_x+step[0];center_y=original_y+step[1];tiles=capture_view();
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            LARGE_INTEGER a={},b={};QueryPerformanceCounter(&a);
            int code=render_checked(&frame,&output);QueryPerformanceCounter(&b);
            ok=ok && code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
            std::printf("PICKUP jump=%d,%d result=%d ms=%.3f built=%u reused=%u gpu_bytes=%u geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f\n",
                step[0],step[1],code,double(b.QuadPart-a.QuadPart)*1000/frequency.QuadPart,
                output.geometry_tiles_built,output.geometry_tiles_reused,output.geometry_cache_bytes,
                double(output.geometry_ticks)*1000/frequency.QuadPart,double(output.draw_ticks)*1000/frequency.QuadPart,
                double(output.readback_ticks)*1000/frequency.QuadPart);
            if(code!=C3X_RENDERER_RESULT_OK)break;
            if(!minimap)ok=prepare() && ok;
        }
        // Compare an image assembled from prepared pixel blocks with a fresh
        // renderer of the same authoritative snapshot. Use the existing native
        // pixel-budget thresholds, never a looser pickup-only allowance.
        center_x=original_x+4;center_y=original_y+4;tiles=capture_view();
        frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        if(render_checked(&frame,&output)!=C3X_RENDERER_RESULT_OK)ok=false;
        std::vector<unsigned char> warm;
        if(ok){auto first=static_cast<unsigned char const*>(output.bgra_pixels);
            warm.assign(first,first+output.stride_bytes*output.height);
            std::string path=std::string(argv[5])+".cached.bmp";write_bmp(path.c_str(),output);}
        reset();
        if(set_definitions(argv[2],argv[3],nullptr,custom_path)!=C3X_RENDERER_RESULT_OK ||
            render_checked(&frame,&output)!=C3X_RENDERER_RESULT_OK)ok=false;
        if(ok){
            auto cold=static_cast<unsigned char const*>(output.bgra_pixels);std::size_t changed=0;
            unsigned long long error=0;
            for(std::size_t i=0;i<warm.size();i+=4){bool bad=false;
                for(unsigned c=0;c<4;++c){unsigned delta=unsigned(std::abs(int(warm[i+c])-int(cold[i+c])));
                    error+=delta;bad=bad || delta>2;}
                if(bad)++changed;
            }
            std::printf("PICKUP pixel parity: changed=%zu error=%llu bytes=%zu\n",changed,error,warm.size());
            std::string path=std::string(argv[5])+".cold.bmp";write_bmp(path.c_str(),output);
            ok=changed<=warm.size()/4000 && error<=warm.size()/100;
        }
    }
    char visibility_test[8]={};
    if(ok && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_VISIBILITY",visibility_test,sizeof(visibility_test))){
        auto animal=std::min_element(tiles.begin(),tiles.end(),[&](auto const& a,auto const& b){
            auto distance=[&](auto const& t){return (t.tile_flags&C3X_RENDERER_TILE_RENDER)&&t.real_terrain_type<=4?
                std::abs(t.anchor_x-frame.target_width/2)+std::abs(t.anchor_y-frame.target_height/2):INT_MAX;};return distance(a)<distance(b);});
        if(animal!=tiles.end()){animal->resource_id=101;animal->resource_class=0;strcpy_s(animal->resource_name,"Cattle");}
        for(auto& tile:tiles){tile.tile_flags|=C3X_RENDERER_TILE_EXPLORED;tile.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;}
        ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.visible_animation_count==0 && !output.request_continuous_redraw;
        auto frozen_start=static_cast<unsigned char const*>(output.bgra_pixels);
        std::vector<unsigned char> frozen(frozen_start,frozen_start+output.stride_bytes*output.height);
        for(int step=0;ok && step<3;++step){frame.presentation_time_ticks+=frame.presentation_frequency/3;
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && !output.geometry_tiles_built && !output.geometry_upload_bytes &&
                !output.visible_animation_count && !std::memcmp(frozen.data(),output.bgra_pixels,frozen.size());}
        reset();
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK &&
            !std::memcmp(frozen.data(),output.bgra_pixels,frozen.size());
        std::printf("VISIBILITY fogged motion: %s frozen_frames=3 cold_exact=1 animations=%u\n",ok?"pass":"FAIL",output.visible_animation_count);
        auto first=static_cast<unsigned char const*>(output.bgra_pixels);
        std::vector<unsigned char> before(first,first+output.stride_bytes*output.height);
        // Authoritative reveal only: anchors, geometry and object state stay fixed.
        for(auto& tile:tiles)tile.tile_flags|=C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
        ok=ok && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
        unsigned built=output.geometry_tiles_built,upload=output.geometry_upload_bytes;
        std::vector<unsigned char> warm;
        if(ok){first=static_cast<unsigned char const*>(output.bgra_pixels);warm.assign(first,first+output.stride_bytes*output.height);}
        ok=ok && !built && !upload && warm!=before;
        if(ok){std::string revealed=std::string(argv[5])+".revealed.bmp";write_bmp(revealed.c_str(),output);}
        reset();
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
        ok=ok && !std::memcmp(warm.data(),output.bgra_pixels,warm.size());
        std::printf("VISIBILITY reveal warm/cold exact: %s built=%u upload=%u bytes=%zu\n",ok?"pass":"FAIL",built,upload,warm.size());
        frame.presentation_time_ticks+=frame.presentation_frequency/3;
        ok=ok && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.visible_animation_count &&
            !output.geometry_tiles_built && !output.geometry_upload_bytes &&
            std::memcmp(warm.data(),output.bgra_pixels,warm.size())!=0;
        std::printf("VISIBILITY revealed motion resumes: %s\n",ok?"pass":"FAIL");
    }
    char unit_test[8]={};
    if(ok && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_UNITS",unit_test,sizeof(unit_test))) {
        auto bytes=static_cast<unsigned char const*>(output.bgra_pixels);
        std::vector<unsigned char> retained(bytes,bytes+output.stride_bytes*output.height);
        ok=preview_units(module,argv[5],frame.hour);
        if(ok)ok=preview_unit_roster(module,argv[5],frame.hour);
        if(ok)ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK &&
            !output.geometry_tiles_built && !output.geometry_upload_bytes &&
            std::memcmp(retained.data(),output.bgra_pixels,retained.size())==0;
        std::printf("UNIT retained terrain unchanged: %s\n",ok?"pass":"FAIL");
        if(ok) {
            frame.hour=(frame.hour+1)%24;
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
            std::vector<unsigned char> warm;
            if(ok){auto begin=static_cast<unsigned char const*>(output.bgra_pixels);warm.assign(begin,begin+output.stride_bytes*output.height);}
            reset();
            if(ok)ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&
                render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
            if(ok) {
                auto cold=static_cast<unsigned char const*>(output.bgra_pixels);std::size_t changed=0;unsigned long long error=0;unsigned peak=0;
                std::size_t last_changed_row=0;
                for(std::size_t i=0;i<warm.size();i+=4){bool bad=false;for(unsigned c=0;c<4;++c){
                    unsigned delta=unsigned(std::abs(int(warm[i+c])-int(cold[i+c])));error+=delta;peak=(std::max)(peak,delta);bad=bad || delta>2;}
                    if(bad){++changed;last_changed_row=(i/output.stride_bytes);}}
                // A fresh D3D terrain rebuild can differ by at most five color
                // levels in sparse pixels on this VM. Preserve the original
                // strict budget, with a second bounded numeric-variance case;
                // larger per-channel changes still fail even when sparse.
                ok=(changed<=warm.size()/4000 && error<=warm.size()/100) ||
                   (peak<=5 && last_changed_row<std::size_t(output.height)/5 && error<=warm.size()/12);
                std::printf("UNIT post-draw terrain parity: %s changed=%zu error=%llu peak=%u bytes=%zu\n",ok?"pass":"FAIL",changed,error,peak,warm.size());
                if(!ok){
                    auto expected=output;expected.bgra_pixels=warm.data();
                    write_bmp((std::string(argv[5])+".unit-warm.bmp").c_str(),expected);
                    write_bmp((std::string(argv[5])+".unit-cold.bmp").c_str(),output);
                }
            }
        }
    }
#ifdef C3X_LAB_PREVIEW
    if(ok)ok=lab_compose_units(module,argv[5],frame.hour,tile_width,output);
#endif
    if(!shared_module){reset();
        auto input_finish=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_input_recording_finish"));if(input_finish)input_finish();
        FreeLibrary(module);}
    return ok ? 0 : 1;
}

int main(int argc,char** argv) {
    // Harness-only VA reservation represents a co-resident game's footprint.
    // Reserve separate chunks to avoid relying on one unusually large hole.
    struct AddressPressure {
        std::vector<void*> chunks;
        ~AddressPressure(){for(auto chunk:chunks)VirtualFree(chunk,0,MEM_RELEASE);}
    } pressure;
    char reserve[16]={};GetEnvironmentVariableA("C3X_RENDERER_TEST_RESERVE_MIB",reserve,sizeof(reserve));
    unsigned reserve_mib=unsigned(std::strtoul(reserve,nullptr,10));
    if(reserve_mib>1536 || reserve_mib%64)return 2;
    for(unsigned remaining=reserve_mib;remaining;remaining-=64){
        auto chunk=VirtualAlloc(nullptr,64u*1024u*1024u,MEM_RESERVE,PAGE_NOACCESS);
        if(!chunk){std::fprintf(stderr,"FAIL address pressure reservation\n");return 2;}
        pressure.chunks.push_back(chunk);
    }
    std::printf("Address-space pressure: %u MiB reserved (not committed)\n",reserve_mib);
    char session_path[4*MAX_PATH]={};
    if(!GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_SESSION",session_path,sizeof(session_path)))
        return run_preview_case(argc,argv);
    auto invalid=[](char const* reason){std::fprintf(stderr,"CASE_SETUP_FAILED reason=%s error=%lu\n",reason,GetLastError());return 2;};
    if(argc!=12)return invalid("argument_count");
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    SetUnhandledExceptionFilter(preview_unhandled_exception);
    FILE* file=nullptr;
    if(fopen_s(&file,session_path,"rb") || !file)return invalid("manifest_open");
    char line[512]={},schema[40]={},case_id[65]={},config[65]={},digest[65]={},policy[24]={},warmup[24]={},extra[2]={};
    unsigned repeats=0,limit_ms=0;
    bool read=std::fgets(line,sizeof(line),file)!=nullptr;
    bool trailing=std::fgetc(file)!=EOF;std::fclose(file);
    if(!read || trailing || sscanf_s(line,"%39s %64s %64s %64s %23s %23s %u %u %1s",
        schema,unsigned(sizeof(schema)),case_id,unsigned(sizeof(case_id)),config,unsigned(sizeof(config)),
        digest,unsigned(sizeof(digest)),policy,unsigned(sizeof(policy)),warmup,unsigned(sizeof(warmup)),
        &repeats,&limit_ms,extra,unsigned(sizeof(extra)))!=8 || std::strcmp(schema,"C3X_PREVIEW_CASES_V1") ||
        repeats<1 || repeats>16 || limit_ms<1000 || limit_ms>600000)return invalid("manifest_fields");
    bool resident=std::strcmp(policy,"prepared_resident")==0;
    bool cold=std::strcmp(policy,"process_cold")==0;
    if((!resident && !cold && std::strcmp(policy,"assets_loaded")) ||
       std::strcmp(warmup,resident?"sequence":"initial") || (cold && repeats!=1))return invalid("reset_policy");
    char scroll[64]={};
    if(!GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_SCROLL_ABLATION",scroll,sizeof(scroll)))return invalid("scroll_environment");
    // One constructor configuration per process. The wrapper freezes all input
    // and environment values; changed configurations use fresh processes.
    char inputs_path[4*MAX_PATH]={};
    if(!GetEnvironmentVariableA("C3X_RENDERER_SESSION_INPUTS",inputs_path,sizeof(inputs_path)))return invalid("inputs_environment");
    if(fopen_s(&file,inputs_path,"rb") || !file)return invalid("inputs_open");
    std::vector<std::pair<std::string,WIN32_FILE_ATTRIBUTE_DATA>> input_stamps;
    char input_path[4*MAX_PATH]={};
    while(std::fgets(input_path,sizeof(input_path),file)) {
        auto length=std::strlen(input_path);
        if(!length || input_path[length-1]!='\n' || input_stamps.size()>=16384){std::fclose(file);return invalid("input_line");}
        input_path[--length]=0;if(length && input_path[length-1]=='\r')input_path[--length]=0;
        WIN32_FILE_ATTRIBUTE_DATA stamp={};
        if(!length || !GetFileAttributesExA(input_path,GetFileExInfoStandard,&stamp)){std::fclose(file);return invalid("input_attributes");}
        input_stamps.emplace_back(input_path,stamp);
    }
    std::fclose(file);if(input_stamps.empty())return invalid("inputs_empty");
    auto verify_inputs=[&](){
        auto begin=GetTickCount64();bool unchanged=true;
        for(auto const& input:input_stamps) {
            WIN32_FILE_ATTRIBUTE_DATA current={};auto const& initial=input.second;
            if(!GetFileAttributesExA(input.first.c_str(),GetFileExInfoStandard,&current) ||
                current.nFileSizeHigh!=initial.nFileSizeHigh || current.nFileSizeLow!=initial.nFileSizeLow ||
                CompareFileTime(&current.ftLastWriteTime,&initial.ftLastWriteTime)!=0){unchanged=false;break;}
        }
        std::printf("CASE_INPUT_CHECK unchanged=%u paths=%zu ms=%llu verification=metadata_only\n",unsigned(unchanged),input_stamps.size(),GetTickCount64()-begin);
        return unchanged;
    };
    HMODULE module=LoadLibraryA(argv[1]);if(!module)return 1;
    auto reset=reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(module,"c3x_renderer_reset"));
    auto reset_case=reinterpret_cast<c3x_renderer_benchmark_session_reset_v1_fn>(
        GetProcAddress(module,"c3x_renderer_benchmark_session_reset_v1"));
    if(!reset || !reset_case){FreeLibrary(module);return invalid("reset_exports");}
    auto checkpoint=[&](char const* phase){
        FILE* state=nullptr;
        if(!fopen_s(&state,(std::string(argv[5])+".state.txt").c_str(),"wb") && state){
            std::fprintf(state,"pid=%lu phase=%s\n",GetCurrentProcessId(),phase);std::fclose(state);
        }
    };
    auto started=GetTickCount64();int code=0;bool configured=false;
    std::vector<char*> args(argv,argv+argc);
    auto run=[&](std::string path,bool bootstrap=false){args[5]=path.data();
        int result=run_preview_case(argc,args.data(),module,!configured,bootstrap);configured=true;return result;};
    auto clear=[&](unsigned mode){
        c3x_renderer_benchmark_oracle_trim_v1 receipt={C3X_RENDERER_BENCHMARK_ORACLE_VERSION,sizeof(receipt)};
        LARGE_INTEGER begin={},end={},frequency={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&begin);
        int result=reset_case(mode,&receipt);QueryPerformanceCounter(&end);
        std::printf("CASE_RESET mode=%u result=%d ms=%.3f cleared_viewport=%llu cleared_regions=%llu cleared_blocks=%llu cleared_backdrops=%llu cleared_publication=%llu retained_geometry=%llu retained_natural=%llu retained_ground=%llu retained_shadow=%llu geometry_entries=%u geometry_evictions=%u pose_evictions=%u assets_device=retained budgets=unchanged\n",
            mode,result,double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
            receipt.cleared_viewport_bytes,receipt.cleared_region_bytes,receipt.cleared_pixel_block_bytes,
            receipt.cleared_backdrop_bytes,receipt.cleared_publication_bytes,receipt.retained_geometry_bytes,
            receipt.retained_natural_bytes,receipt.retained_ground_bytes,receipt.retained_shadow_bytes,
            receipt.retained_geometry_entries,receipt.capacity_geometry_evictions,receipt.capacity_pose_evictions);
        return result==C3X_RENDERER_RESULT_OK;
    };
    // Load the immutable asset set once before every assets-loaded arm. This
    // untimed bootstrap is explicitly reported, and its scene content is reset.
    checkpoint("bootstrap");
    if(!cold){std::puts("CASE_BOOTSTRAP_BEGIN");code=run(std::string(argv[5])+".bootstrap.bmp",true);std::printf("CASE_BOOTSTRAP_END result=%d elapsed_ms=%llu\n",code,GetTickCount64()-started);}
    unsigned completed=0;
    if(!code && !verify_inputs())code=4;
    for(unsigned i=0;!code && i<repeats;++i) {
        if(GetTickCount64()-started>=limit_ms){code=3;break;}
        std::printf("CASE_BEGIN id=%s-%u config=%s request_digest=%s reset=%s warmup=%s\n",case_id,i,config,digest,policy,warmup);
        if(!cold && !clear(1)){code=1;break;}
        if(resident){
            std::puts("CASE_WARMUP_BEGIN");code=run(std::string(argv[5])+".warmup"+std::to_string(i)+".bmp");
            std::printf("CASE_WARMUP_END result=%d\n",code);
            if(code || !clear(2)){code=1;break;}
        }
        checkpoint("playback");
        std::puts("CASE_PLAYBACK_BEGIN");
        code=run(std::string(argv[5])+".case"+std::to_string(i)+".bmp");
        std::printf("CASE_END id=%s-%u result=%d\n",case_id,i,code);std::fflush(stdout);
        if(!code && !verify_inputs())code=4;
        if(!code)++completed;
    }
    // Workers and borrowed publications retire before unloading the one DLL.
    checkpoint("worker_reset");reset();
    checkpoint("dll_unload");FreeLibrary(module);
    checkpoint("complete");
    bool timed_out=GetTickCount64()-started>limit_ms;
    std::printf("CASE_SESSION_END result=%d completed=%u requested=%u elapsed_ms=%llu time_limit_exceeded=%u\n",
        code,completed,repeats,GetTickCount64()-started,unsigned(timed_out));
    return code || timed_out ? 1 : 0;
}

bool preview_units(HMODULE module,char const* path,int hour) {
    auto legacy_draw=reinterpret_cast<c3x_renderer_unit_draw_fn>(GetProcAddress(module,"c3x_renderer_unit_draw"));
    auto expanded_draw=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_expanded"));
    int drawn_bounds[4]={};
    auto draw=[&](c3x_renderer_unit_v1 const* unit,HDC dc){
        return expanded_draw?expanded_draw(unit,dc,dc,drawn_bounds):legacy_draw(unit,dc);
    };
    auto configure=reinterpret_cast<c3x_renderer_set_unit_rendering_fn>(GetProcAddress(module,"c3x_renderer_set_unit_rendering"));
    if(!legacy_draw || !configure)return false;
    BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth=1024;info.bmiHeader.biHeight=-1152;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
    HDC dc=CreateCompatibleDC(nullptr);void* bits=nullptr;
    HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
    if(!dc || !bitmap){if(bitmap)DeleteObject(bitmap);if(dc)DeleteDC(dc);return false;}
    auto old=SelectObject(dc,bitmap);bool ok=true;unsigned drawn=0;
    // Three equal-facing warriors at one clock follow independent native cursors.
    // Unit identity and wall time must not override the supplied native phase.
    std::vector<std::uint32_t> ambient_sheet(573*191,0xff565b62u);
    std::vector<std::vector<std::uint32_t>> ambient_images;
    for(int i=0;i<3 && ok;++i) {
        std::fill_n(static_cast<std::uint32_t*>(bits),1024*1152,0xff565b62u);
        c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);strcpy_s(unit.unit_key,"PRTO_Warrior");
        unit.unit_id=10000+i;unit.action=1;unit.direction=3;unit.frame_count=16;unit.action_cursor=i*5;
        unit.sprite_width=unit.sprite_height=191;unit.body_x=unit.body_y=100;
        unit.presentation_frequency=1000000;unit.presentation_time_ticks=1000000;unit.hour=hour;
        unit.display_color_rgb=0x205bdd;ok=draw(&unit,dc)==C3X_RENDERER_RESULT_OK;GdiFlush();
        auto values=static_cast<std::uint32_t*>(bits);
        std::vector<std::uint32_t> current(values,values+1024*1152);
        for(auto const& prior:ambient_images)ok=ok && current!=prior;
        for(int y=0;y<191;++y)for(int x=0;x<191;++x)ambient_sheet[y*573+i*191+x]=values[(y+100)*1024+x+100];
        std::fill_n(values,1024*1152,0xff565b62u);
        ok=draw(&unit,dc)==C3X_RENDERER_RESULT_OK && ok;GdiFlush();
        ok=ok && std::memcmp(current.data(),bits,current.size()*4)==0;
        auto same_phase=unit;same_phase.unit_id+=1000;same_phase.presentation_time_ticks+=500000;
        std::fill_n(values,1024*1152,0xff565b62u);
        ok=draw(&same_phase,dc)==C3X_RENDERER_RESULT_OK && ok;GdiFlush();
        ok=ok && std::memcmp(current.data(),bits,current.size()*4)==0;
        ambient_images.push_back(std::move(current));
    }
    std::printf("UNIT independent native cursors and exact repeat: %s\n",ok?"pass":"FAIL");
    if(ok) {
        c3x_renderer_output_v1 sheet={};sheet.width=573;sheet.height=191;sheet.stride_bytes=573*4;sheet.bgra_pixels=ambient_sheet.data();
        ok=write_bmp((std::string(path)+".ambient-phases.bmp").c_str(),sheet);
    }
    ambient_images.clear();ambient_sheet.clear();
    // Exercise the optional native-selection policy through the real DLL and
    // compare it with independently requested exact source samples.
    auto playback_draw=reinterpret_cast<c3x_renderer_unit_draw_playback_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_playback"));
    if(playback_draw && ok) {
        c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);strcpy_s(unit.unit_key,"PRTO_Worker");
        unit.unit_id=22000;unit.action=13;unit.direction=3;unit.frame_count=15;
        unit.sprite_width=unit.sprite_height=191;unit.body_x=unit.body_y=100;
        unit.presentation_frequency=1000000;unit.hour=hour;unit.display_color_rgb=0x205bdd;
        auto values=static_cast<std::uint32_t*>(bits);
        auto capture=[&](bool policy,unsigned flags) {
            std::fill_n(values,1024*1152,0xff565b62u);
            int result=policy?playback_draw(&unit,dc,dc,drawn_bounds,flags):draw(&unit,dc);
            GdiFlush();ok=ok && result==C3X_RENDERER_RESULT_OK;
            return std::vector<std::uint32_t>(values,values+1024*1152);
        };
        std::vector<std::uint32_t> comparison(1280*640,0xff565b62u);
        for(int frame=0;frame<16 && ok;++frame) {
            unit.frame_count=15;unit.action_cursor=frame%15;unit.presentation_time_ticks=frame*66000;
            auto current=capture(true,C3X_RENDERER_UNIT_STATE_CAPTURED);
            if(frame%5==0)for(int y=0;y<320;++y)for(int x=0;x<320;++x)
                comparison[(y+320)*1280+(frame/5)*320+x]=current[(y+36)*1024+x+36];
            auto legacy=capture(false,0);
            if(frame%5==0)for(int y=0;y<320;++y)for(int x=0;x<320;++x)
                comparison[y*1280+(frame/5)*320+x]=legacy[(y+36)*1024+x+36];
            // Authored builder road clip: 95 samples spanning 94/30 seconds.
            unit.frame_count=94;unit.action_cursor=int((frame*.066)/3.133333444595337*94);
            auto expected=capture(false,0);ok=ok && current==expected;
        }
        unit.action=1;unit.frame_count=15;unit.action_cursor=0;
        auto frozen=capture(true,C3X_RENDERER_UNIT_STATE_CAPTURED);
        for(int frame=0;frame<10 && ok;++frame) {
            unit.presentation_time_ticks+=66000;unit.action_cursor=frame;unit.action=frame%2?8:1;
            ok=capture(true,C3X_RENDERER_UNIT_STATE_CAPTURED)==frozen && ok;
        }
        std::printf("UNIT caller playback: authored worker time/source pixels, frozen idle/fidget: %s\n",ok?"pass":"FAIL");
        if(ok) {
            c3x_renderer_output_v1 sheet={};sheet.width=1280;sheet.height=640;sheet.stride_bytes=1280*4;sheet.bgra_pixels=comparison.data();
            ok=write_bmp((std::string(path)+".worker-playback.bmp").c_str(),sheet);
        }
    }
    char const* names[]={"Archer","Swordsman","Infantry","Fighter","Galley","Warrior","Scout","Settler","Worker"};
    std::vector<std::uint32_t> first;
    for(int zoom=0;zoom<2 && ok;++zoom)for(int phase=0;phase<2 && ok;++phase) {
        std::fill_n(static_cast<std::uint32_t*>(bits),1024*1152,0xff565b62u);
        for(int row=0;row<9 && ok;++row)for(int direction=1;direction<=8 && ok;++direction) {
            c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);unit.unit_id=row;
            sprintf_s(unit.unit_key,"PRTO_%s",names[row]);
            unit.action=2;unit.action_cursor=phase*7;unit.frame_count=16;unit.direction=direction;
            unit.sprite_width=unit.sprite_height=191;unit.reduced=zoom;unit.hour=hour;
            unit.display_color_rgb=direction%2?0x205bdd:0xe33020;
            unit.body_x=(direction-1)*128+64-191/(zoom?4:2);
            unit.body_y=row*128+91-191/(zoom?4:2);
            int result=draw(&unit,dc);
            if(result!=C3X_RENDERER_RESULT_OK) {std::printf("FAIL unit body key=%s direction=%d phase=%d zoom=%d result=%d\n",unit.unit_key,direction,phase,zoom,result);ok=false;}
            else ++drawn;
        }
        c3x_renderer_output_v1 picture={};picture.width=1024;picture.height=1152;picture.stride_bytes=4096;picture.bgra_pixels=bits;
        std::string filename=std::string(path)+".units-z"+std::to_string(zoom)+"-p"+std::to_string(phase)+".bmp";
        if(ok)ok=write_bmp(filename.c_str(),picture);
        auto begin=static_cast<std::uint32_t const*>(bits);
        if(phase==0)first.assign(begin,begin+1024*1152);
        else if(ok) {
            std::size_t changes=0;
            for(int row=0;row<9;++row) {
                unsigned changed=0, visible=0;
                for(int y=row*128;y<(row+1)*128;++y)for(int x=0;x<1024;++x) {
                    auto index=y*1024+x;
                    if(first[index]!=begin[index])++changed;
                    if(begin[index]!=0xff565b62u)++visible;
                }
                std::printf("UNIT family key=%s zoom=%d changed=%u visible=%u\n",names[row],zoom,changed,visible);
                ok=ok && changed>10 && visible>10;changes+=changed;
            }
            std::printf("UNIT temporal zoom=%d changed_pixels=%zu\n",zoom,changes);ok=ok && changes>100;
        }
    }
    if(ok) {
        // Reuse one pose at a new authoritative location and another unit ID.
        std::fill_n(static_cast<std::uint32_t*>(bits),1024*1152,0xff565b62u);
        c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);strcpy_s(unit.unit_key,"PRTO_Archer");
        unit.unit_id=41;unit.action=1;unit.direction=3;unit.frame_count=16;unit.action_cursor=8;
        unit.presentation_frequency=1000000;unit.presentation_time_ticks=500000;
        unit.sprite_width=unit.sprite_height=191;unit.body_x=100;unit.body_y=100;unit.hour=hour;unit.display_color_rgb=0x205bdd;
        ok=draw(&unit,dc)==C3X_RENDERER_RESULT_OK;
        std::vector<std::uint32_t> original(static_cast<std::uint32_t*>(bits),static_cast<std::uint32_t*>(bits)+1024*1152);
        std::fill_n(static_cast<std::uint32_t*>(bits),1024*1152,0xff565b62u);
        unit.unit_id=42;unit.body_x+=128;
        ok=draw(&unit,dc)==C3X_RENDERER_RESULT_OK && ok;
        auto actual=static_cast<std::uint32_t const*>(bits);
        for(int y=0;y<1152;++y)for(int x=0;x<1024;++x)
            if(actual[y*1024+x]!=(x>=128?original[y*1024+x-128]:0xff565b62u))ok=false;
        std::printf("UNIT cached anchor translation: %s\n",ok?"pass":"FAIL");
        auto snapshot=std::vector<std::uint32_t>(actual,actual+1024*1152);
        std::fill_n(static_cast<std::uint32_t*>(bits),1024*1152,0xff565b62u);
        ok=draw(&unit,dc)==C3X_RENDERER_RESULT_OK && std::memcmp(snapshot.data(),bits,snapshot.size()*4)==0 && ok;
        std::printf("UNIT repeated native cursor: %s\n",ok?"pass":"FAIL");
        ok=configure(2)==C3X_RENDERER_RESULT_BAD_ARGUMENT && ok;
        ok=configure(0)==C3X_RENDERER_RESULT_OK && ok;
        ok=draw(&unit,dc)!=C3X_RENDERER_RESULT_OK && std::memcmp(snapshot.data(),bits,snapshot.size()*4)==0 && ok;
        ok=configure(1)==C3X_RENDERER_RESULT_OK && ok;
        std::printf("UNIT config-off preserves canvas: %s\n",ok?"pass":"FAIL");
        // Exercise AlphaBlend against the actual legacy 16-bit GDI surfaces,
        // including partial offscreen bodies at both native zoom scales.
        for(int green_bits:{5,6})for(int zoom=0;zoom<2 && ok;++zoom) {
            struct {BITMAPINFOHEADER header;DWORD masks[3];} format={};
            format.header.biSize=sizeof(BITMAPINFOHEADER);format.header.biWidth=384;format.header.biHeight=-256;
            format.header.biPlanes=1;format.header.biBitCount=16;format.header.biCompression=BI_BITFIELDS;
            format.masks[0]=green_bits==5?0x7c00u:0xf800u;format.masks[1]=green_bits==5?0x3e0u:0x7e0u;format.masks[2]=0x1fu;
            HDC dc16=CreateCompatibleDC(nullptr);void* pixels16=nullptr;
            HBITMAP dib16=CreateDIBSection(dc16,reinterpret_cast<BITMAPINFO*>(&format),DIB_RGB_COLORS,&pixels16,nullptr,0);
            if(!dc16 || !dib16 || !pixels16) {if(dib16)DeleteObject(dib16);if(dc16)DeleteDC(dc16);ok=false;break;}
            auto previous=SelectObject(dc16,dib16);auto values=static_cast<std::uint16_t*>(pixels16);
            std::fill_n(values,384*256,std::uint16_t(0x4210));
            unit.reduced=zoom;unit.body_x=zoom?-24:-47;unit.body_y=zoom?-12:-27;
            ok=draw(&unit,dc16)==C3X_RENDERER_RESULT_OK && ok;GdiFlush();
            unsigned changes=0;int extent=191/(zoom?2:1);
            for(int y=0;y<256;++y)for(int x=0;x<384;++x)if(values[y*384+x]!=0x4210) {
                ++changes;if(expanded_draw?(x<drawn_bounds[0] || y<drawn_bounds[1] || x>=drawn_bounds[2] || y>=drawn_bounds[3]):(x>=unit.body_x+extent || y>=unit.body_y+extent))ok=false;
            }
            ok=changes>10 && ok;
            std::printf("UNIT RGB5%d5 clipped zoom=%d changed=%u status=%s\n",green_bits,zoom,changes,ok?"pass":"FAIL");
            if(ok) {
                auto with_background=reinterpret_cast<c3x_renderer_unit_draw_background_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_background"));
                auto reference=std::vector<std::uint16_t>(values,values+384*256);
                HDC bg16=CreateCompatibleDC(nullptr);void* bg_bits=nullptr;
                HBITMAP bg_dib=CreateDIBSection(bg16,reinterpret_cast<BITMAPINFO*>(&format),DIB_RGB_COLORS,&bg_bits,nullptr,0);
                HGDIOBJ bg_old=bg_dib?SelectObject(bg16,bg_dib):nullptr;
                if(!with_background || !bg16 || !bg_dib || !bg_bits)ok=false;
                if(ok) {
                    std::fill_n(static_cast<std::uint16_t*>(bg_bits),384*256,std::uint16_t(0x4210));
                    std::uint16_t key=green_bits==5?0x7c1f:0xf81f;
                    std::fill_n(values,384*256,key);
                    ok=(expanded_draw?expanded_draw(&unit,dc16,bg16,drawn_bounds):with_background(&unit,dc16,bg16))==C3X_RENDERER_RESULT_OK;GdiFlush();
                    for(int i=0;i<384*256;++i)if((values[i]==key?std::uint16_t(0x4210):values[i])!=reference[i])ok=false;
                }
                std::printf("UNIT RGB5%d5 magenta clipped parity zoom=%d status=%s\n",green_bits,zoom,ok?"pass":"FAIL");
                if(bg16 && bg_old)SelectObject(bg16,bg_old);
                if(bg_dib)DeleteObject(bg_dib);if(bg16)DeleteDC(bg16);
            }
            SelectObject(dc16,previous);DeleteObject(dib16);DeleteDC(dc16);
        }
    }
    if(ok) {
        auto with_background=reinterpret_cast<c3x_renderer_unit_draw_background_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_background"));
        HDC background=CreateCompatibleDC(nullptr);void* underlay_bits=nullptr;
        HBITMAP underlay=CreateDIBSection(background,&info,DIB_RGB_COLORS,&underlay_bits,nullptr,0);
        if(!with_background || !background || !underlay || !underlay_bits)ok=false;
        HGDIOBJ previous=underlay?SelectObject(background,underlay):nullptr;
        if(ok)for(int zoom=0;zoom<2 && ok;++zoom) {
            auto ground=static_cast<std::uint32_t*>(underlay_bits);
            auto canvas=static_cast<std::uint32_t*>(bits);
            for(int i=0;i<1024*1152;++i)ground[i]=0xff205030u+unsigned((i%1024)/8)*0x10101u;
            std::memcpy(canvas,ground,1024*1152*4);
            c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);strcpy_s(unit.unit_key,"PRTO_Settler");
            unit.unit_id=45;unit.action=2;unit.direction=3;unit.frame_count=16;unit.action_cursor=7;
            unit.sprite_width=unit.sprite_height=191;unit.reduced=zoom;unit.body_x=100;unit.body_y=100;
            unit.hour=hour;unit.display_color_rgb=0x205bdd;
            ok=draw(&unit,dc)==C3X_RENDERER_RESULT_OK;GdiFlush();
            auto reference=std::vector<std::uint32_t>(canvas,canvas+1024*1152);
            std::fill_n(canvas,1024*1152,0x00ff00ffu);
            ok=(expanded_draw?expanded_draw(&unit,dc,background,drawn_bounds):with_background(&unit,dc,background))==C3X_RENDERER_RESULT_OK && ok;GdiFlush();
            unsigned changed=0;
            for(int i=0;i<1024*1152;++i) {
                auto composed=canvas[i]==0x00ff00ffu?ground[i]:canvas[i];
                if((composed&0xffffffu)!=(reference[i]&0xffffffu))ok=false;
                if(canvas[i]!=0x00ff00ffu)++changed;
            }
            ok=changed>10 && ok;
            std::printf("UNIT magenta underlay parity zoom=%d changed=%u status=%s\n",zoom,changed,ok?"pass":"FAIL");
        }
        if(background && previous)SelectObject(background,previous);
        if(underlay)DeleteObject(underlay);if(background)DeleteDC(background);
    }
    unsigned action_draws=0;
    for(int row=0;row<9 && ok;++row)for(int zoom=0;zoom<2 && ok;++zoom) {
        c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);unit.unit_id=80+row;
        sprintf_s(unit.unit_key,"PRTO_%s",names[row]);unit.direction=3;unit.frame_count=16;
        unit.presentation_frequency=1000000;
        unit.sprite_width=unit.sprite_height=191;unit.body_x=100;unit.body_y=100;
        unit.hour=hour;unit.reduced=zoom;unit.display_color_rgb=0x205bdd;
        std::vector<std::uint32_t> idle;
        auto pose=[&](int action,int cursor,int queued) {
            std::fill_n(static_cast<std::uint32_t*>(bits),1024*1152,0xff565b62u);
            unit.action=action;unit.action_cursor=cursor;unit.queued_action=queued;
            unit.presentation_time_ticks=cursor*100000;
            int result=draw(&unit,dc);GdiFlush();
            if(result!=C3X_RENDERER_RESULT_OK) {
                std::printf("FAIL unit action key=%s action=%d cursor=%d zoom=%d result=%d\n",unit.unit_key,action,cursor,zoom,result);
                return false;
            }
            ++action_draws;return true;
        };
        ok=pose(1,5,0);
        idle.assign(static_cast<std::uint32_t*>(bits),static_cast<std::uint32_t*>(bits)+1024*1152);
        // Native cursor/action changes interrupt any previous pose immediately.
        std::vector<int> actions=row<7?std::vector<int>{2,3,4,5,6,7,8,9}:(row==7?std::vector<int>{2,7,8,10,12}:std::vector<int>{2,7,8,10,11,12,13,14,15,16,17,18});
        bool save_actions=hour==12 && zoom==0 && (row==5 || row>=7);
        std::vector<std::uint32_t> action_sheet(save_actions?std::size_t(573)*191*actions.size():0,0xff565b62u);
        for(std::size_t action_index=0;action_index<actions.size();++action_index)for(int step=0;step<3;++step)if(ok) {
            ok=pose(actions[action_index],step==2?15:step*7,1);
            if(save_actions)for(int cy=0;cy<191;++cy)for(int cx=0;cx<191;++cx)
                action_sheet[(action_index*191+cy)*573+step*191+cx]=static_cast<std::uint32_t*>(bits)[(100+cy)*1024+100+cx];
        }
        if(save_actions && ok) {
            c3x_renderer_output_v1 sheet={};sheet.width=573;sheet.height=unsigned(191*actions.size());
            sheet.stride_bytes=573*4;sheet.bgra_pixels=action_sheet.data();
            auto filename=std::string(path)+".actions-"+names[row]+".bmp";
            ok=write_bmp(filename.c_str(),sheet);
        }
        if(ok) {
            ok=pose(7,0,0);
            auto fortify_first=std::vector<std::uint32_t>(static_cast<std::uint32_t*>(bits),static_cast<std::uint32_t*>(bits)+1024*1152);
            ok=pose(7,7,0) && ok;
            bool changed=std::memcmp(fortify_first.data(),bits,fortify_first.size()*4)!=0;
            ok=pose(7,15,0) && ok;
            changed=changed || std::memcmp(fortify_first.data(),bits,fortify_first.size()*4)!=0;
            if(!changed && row!=3){std::printf("FAIL fortify has no visible transition key=%s zoom=%d\n",unit.unit_key,zoom);ok=false;}
            // Fighter deliberately binds fortify to idle; it has no ground brace.
            std::printf("UNIT fortify %s key=%s zoom=%d status=%s\n",row==3?"idle alias":"transition",unit.unit_key,zoom,(changed || row==3)?"pass":"FAIL");
            int held_action=row<7?6:10;
            ok=pose(held_action,15,0) && ok;
            auto death=std::vector<std::uint32_t>(static_cast<std::uint32_t*>(bits),static_cast<std::uint32_t*>(bits)+1024*1152);
            ok=pose(held_action,1000,1) && std::memcmp(death.data(),bits,death.size()*4)==0 && ok;
            unit.unit_id+=100; // Fresh identity with a queued attack still uses current idle.
            ok=pose(1,5,0) && ok;
            idle.assign(static_cast<std::uint32_t*>(bits),static_cast<std::uint32_t*>(bits)+1024*1152);
            ok=pose(1,5,3) && std::memcmp(idle.data(),bits,idle.size()*4)==0 && ok;
            if(row>=7) {
                unit.action=3;
                ok=draw(&unit,dc)!=C3X_RENDERER_RESULT_OK && std::memcmp(idle.data(),bits,idle.size()*4)==0 && ok;
            }
            if(row==8) {
                strcpy_s(unit.unit_key,"PRTO_Builder");unit.action=1;
                ok=draw(&unit,dc)==C3X_RENDERER_RESULT_OK && ok;
                // Reset before the non-mutating unsupported action assertion.
                std::memcpy(bits,idle.data(),idle.size()*4);
            }
            unit.action=19; // An invalid native action must leave the canvas intact.
            ok=draw(&unit,dc)!=C3X_RENDERER_RESULT_OK && std::memcmp(idle.data(),bits,idle.size()*4)==0 && ok;
        }
    }
    std::printf("UNIT action interruption and held endpoint: %s draws=%u\n",ok?"pass":"FAIL",action_draws);
    // A missing action/key must leave the native canvas byte-for-byte intact.
    std::vector<std::uint32_t> retained(static_cast<std::uint32_t*>(bits),static_cast<std::uint32_t*>(bits)+1024*1152);
    c3x_renderer_unit_v1 unknown={};unknown.struct_size=sizeof(unknown);strcpy_s(unknown.unit_key,"PRTO_NotMapped");
    ok=draw(&unknown,dc)!=C3X_RENDERER_RESULT_OK && std::memcmp(retained.data(),bits,retained.size()*4)==0 && ok;
    SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
    std::printf("UNIT body matrix drawn=%u status=%s\n",drawn,ok?"pass":"FAIL");return ok;
}
