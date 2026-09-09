#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#include "c3x_renderer_api.h"

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
    void* frames[32]={};USHORT count=CaptureStackBackTrace(0,32,frames,nullptr);
    for(USHORT i=0;i<count;++i)preview_fault_address("stack",frames[i]);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

int main(int argc, char ** argv) {
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
    std::vector<CsvTile> source_tiles;
    if (!read_scene(argv[4], map_width, map_height, source_tiles)) {
        std::fprintf(stderr, "error: could not read BIQ terrain scene\n");
        return 1;
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
    HMODULE module = LoadLibraryA(argv[1]);
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
#ifdef C3X_LAB_PREVIEW
    enable_unit_preview=enable_unit_preview || GetEnvironmentVariableA("C3X_LAB_UNIT_STUDY",unit_preview,sizeof(unit_preview))!=0;
#endif
    if(enable_unit_preview &&
       (!set_units || set_units(1)!=C3X_RENDERER_RESULT_OK))return 1;
    if (set_definitions == nullptr || render == nullptr || reset == nullptr ||
        set_definitions(argv[2], argv[3], nullptr, custom_path) != C3X_RENDERER_RESULT_OK)
        return 1;

    char object_option[8]={};bool objects=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_OBJECTS",object_option,sizeof(object_option))!=0;
    char animation_option[8]={};bool animate=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_ANIMATION",animation_option,sizeof(animation_option))!=0;
    std::vector<std::array<int,2>> resource_sites;
    std::vector<std::array<int,2>> city_object_sites;
    auto capture_view = [&]() {
    int center_raw_x = center_x * tile_width / 2;
    int center_raw_y = center_y * tile_height / 2;
    int shift_x = target_width / 2 - tile_width / 2 - center_raw_x;
    int shift_y = target_height / 2 - tile_height / 2 - center_raw_y;
    std::vector<c3x_renderer_tile_v1> tiles;
    for (CsvTile const & source : source_tiles) {
      for (int wrap_copy = -1; wrap_copy <= 1; ++wrap_copy) {
        int render_x = source.x + wrap_copy * map_width;
        int anchor_x = render_x * tile_width / 2 + shift_x;
        int anchor_y = source.y * tile_height / 2 + shift_y;
        int margin=pickup?tile_width*6:96,vertical=pickup?tile_height*6:128;
        if (anchor_x + tile_width < -margin || anchor_x > target_width + margin ||
            anchor_y + tile_height < -vertical || anchor_y > target_height + vertical)
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
        if(pickup && (anchor_x+tile_width < -tile_width*4 || anchor_x>target_width+tile_width*4 ||
            anchor_y+tile_height < -tile_height*4 || anchor_y>target_height+tile_height*4))
            tile.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
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
    if(objects) {
        std::vector<std::size_t> candidates;
        for(std::size_t i=0;i<tiles.size();++i)if(tiles[i].real_terrain_type<=4 &&
            (tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER))candidates.push_back(i);
        std::sort(candidates.begin(),candidates.end(),[&](auto a,auto b){
            auto distance=[&](auto i){auto const& t=tiles[i];return std::abs(t.anchor_x-target_width/2)+std::abs(t.anchor_y-target_height/2);};
            return distance(a)<distance(b);
        });
        char fixed_city_case[80]={};
        if(GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CITY",fixed_city_case,sizeof(fixed_city_case))){
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
        if(candidates.size()>=6){
            auto& city=tiles[candidates[0]];city.city_id=1;city.city_owner_id=1;city.city_size=2;
            city.city_culture_group=0;city.city_era=2;city.city_flags=C3X_RENDERER_CITY_CAPITAL|C3X_RENDERER_CITY_WALLED;
            char city_case[80]={};
            if(GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CITY",city_case,sizeof(city_case))){
                int culture=0,era=0,size=0,capital=0;
                if(sscanf_s(city_case,"%d,%d,%d,%d",&culture,&era,&size,&capital)==4 &&
                    culture>=0 && culture<=4 && era>=0 && era<=3 && size>=0 && size<=2 && capital>=0 && capital<=1){
                    city.city_culture_group=culture;city.city_era=era;city.city_size=size;
                    city.city_flags=capital?C3X_RENDERER_CITY_CAPITAL:0;
                }
            }
            auto& mine=tiles[candidates[1]];mine.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;mine.route_style=2;
            auto& farm=tiles[candidates[2]];farm.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;farm.irrigation_mask=15;farm.route_style=2;
            auto& resource=tiles[candidates[3]];resource.resource_id=1;resource.resource_class=0;strcpy_s(resource.resource_name,"Iron");
            auto& road=tiles[candidates[4]];road.road_mask=15;road.route_style=2;
            auto& rail=tiles[candidates[5]];rail.road_mask=15;rail.railroad_mask=15;rail.route_style=3;
        }
    }
    if (animate) {
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
    return tiles;
    };
    auto tiles=capture_view();
    c3x_renderer_frame_v1 frame = {};
    frame.api_version = C3X_RENDERER_API_VERSION;
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
    if(background_camera && (!camera_begin || !camera_poll)) {
        std::fputs("camera extension exports missing\n",stderr);return 1;
    }
    unsigned camera_case=0;
    auto render_checked = [&](c3x_renderer_frame_v1 const* input, c3x_renderer_output_v1* result) {
        int code=C3X_RENDERER_RESULT_ERROR;
        if(background_camera) {
            LARGE_INTEGER begin={},accepted={},finished={},frequency={};QueryPerformanceFrequency(&frequency);
            c3x_renderer_i64 ticket=0,obsolete=0;
            // Exercise actual in-flight supersession, not only an empty queue.
            // This earlier valid scene has a different environment/pose clock.
            auto earlier=*input;earlier.hour=(earlier.hour+1)%24;
            earlier.presentation_time_ticks+=earlier.presentation_frequency/2;
            if(camera_begin(&earlier,&obsolete)!=C3X_RENDERER_RESULT_PENDING)return int(C3X_RENDERER_RESULT_ERROR);
            Sleep(10);
            QueryPerformanceCounter(&begin);
            code=camera_begin(input,&ticket);QueryPerformanceCounter(&accepted);
            if(code!=C3X_RENDERER_RESULT_PENDING || camera_poll(obsolete,result)!=C3X_RENDERER_RESULT_SUPERSEDED)
                return int(C3X_RENDERER_RESULT_ERROR);
            auto start=GetTickCount64();unsigned polls=0;bool first_image=false;
            ++camera_case;
            while(code==C3X_RENDERER_RESULT_PENDING && GetTickCount64()-start<120000) {
                code=camera_poll(ticket,result);++polls;
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
            std::printf("CAMERA ticket=%lld accepted_ms=%.3f final_ms=%.3f polls=%u stale_rejected=1 result=%d\n",
                static_cast<long long>(ticket),double(accepted.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
                double(finished.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,polls,code);
            std::fflush(stdout);
        }else code=render(input,result);
        return code==C3X_RENDERER_RESULT_OK && !preview_ownership(*input,*result)
            ? int(C3X_RENDERER_RESULT_ERROR) : code;
    };
    int result = render_checked(&frame, &output);
    std::size_t expected_rendered = 0;
    for (c3x_renderer_tile_v1 const & tile : tiles)
        if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) != 0)
            ++expected_rendered;
    bool ok = result == C3X_RENDERER_RESULT_OK &&
              (pickup ? output.rendered_tile_count >= expected_rendered : output.rendered_tile_count == expected_rendered) &&
              output.fallback_tile_count == 0 && write_bmp(argv[5], output);
#ifdef C3X_LAB_PREVIEW
    if(ok)ok=lab_verify_objects(frame,output);
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
    char cycle_option[16]={};
    int camera_cycles=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_CYCLES",cycle_option,sizeof(cycle_option))?
        std::clamp(std::atoi(cycle_option),2,10):2;
    // Sample outside the timed interaction. Free VA is not free physical RAM;
    // the largest available region also exposes fragmentation in this x86 host.
    auto camera_memory = [&]() {
        MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);
        if(!GlobalMemoryStatusEx(&memory))return;
        std::uintptr_t address=0;SIZE_T largest=0;MEMORY_BASIC_INFORMATION region={};
        while(VirtualQuery(reinterpret_cast<void const*>(address),&region,sizeof(region))){
            if(region.State==MEM_FREE)largest=(std::max)(largest,region.RegionSize);
            auto next=reinterpret_cast<std::uintptr_t>(region.BaseAddress)+region.RegionSize;
            if(next<=address)break;
            address=next;
        }
        std::printf("CAMERA memory available_virtual=%llu largest_free_region=%zu total_virtual=%llu\n",
            memory.ullAvailVirtual,largest,memory.ullTotalVirtual);
    };
    if(ok && zoom_benchmark) {
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
        std::vector<std::vector<unsigned char>> reference(5);
        int levels[]={128,96,64,192,160}; // Native Z cycle, normal in the middle.
        for(int cycle=0;cycle<camera_cycles && ok;++cycle)for(int level=0;level<5 && ok;++level){
            tile_width=levels[level];tile_height=tile_width/2;tiles=capture_view();
            frame.tile_width=tile_width;frame.tile_height=tile_height;
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            int code=render_checked(&frame,&output);QueryPerformanceCounter(&end);
            ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
            std::printf("ZOOM cycle=%d width=%d result=%d tiles=%u built=%u reused=%u cache_bytes=%u recoveries=%u ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f\n",
                cycle,tile_width,code,output.rendered_tile_count,output.geometry_tiles_built,output.geometry_tiles_reused,
                output.geometry_cache_bytes,output.device_recoveries,double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,
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
                ok=changed<=count/4000 && error<=count/100;
                std::printf("ZOOM parity width=%d changed=%zu error=%llu status=%s\n",tile_width,changed,error,ok?"pass":"FAIL");
            }
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
    if(ok && animate && !zoom_benchmark && !navigation_benchmark) {
        // Exercise animation after an immutable viewport LRU restore, not only
        // after the unchanged-current-view fast path.
        auto initial=static_cast<unsigned char const*>(output.bgra_pixels);
        std::vector<unsigned char> initial_pixels(initial,initial+std::size_t(output.stride_bytes)*output.height);
        int original_width=tile_width;
        for(int next:{original_width==112?96:112,original_width}) {
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
            bool same=changed<=cached.size()/4000 && error<=cached.size()/100;
            std::printf("ANIMATION %s parity: %s changed=%zu error=%llu bytes=%zu\n",label,same?"pass":"FAIL",changed,error,cached.size());
            return same;
        };
        if(ok) {
            center_x+=4;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.visible_animation_count>0;
            if(ok)ok=compare_cold("scroll");
        }
        if(ok) {
            for(auto & tile:tiles){tile.resource_id=-1;tile.resource_name[0]='\0';}
            ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.visible_animation_count==0 && !output.request_continuous_redraw;
            if(ok)ok=compare_cold("removal");
        }
    }
    char color_option[8]={};
    if(ok && GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_COLOR",color_option,sizeof(color_option)))
        ok=write_color_preview(argv[5],module,output);
    if(ok && objects){
        unsigned ownership=0;for(unsigned i=0;i<output.replacement_tile_count;++i)ownership|=output.replacement_tile_flags[i];
        unsigned expected=C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED|C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED|
            C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED|C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED|
            C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED|C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED;
        ok=(ownership&expected)==expected;
        std::printf("PICKUP synthetic objects: %s ownership=%u\n",ok?"pass":"FAIL",ownership);
    }
    std::printf("BIQ %dx%d viewport: %u visible tiles, %u fallback, output=%s\n",
                map_width, map_height, output.rendered_tile_count, output.fallback_tile_count, argv[5]);
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
                auto cold=static_cast<unsigned char const*>(output.bgra_pixels);std::size_t changed=0;unsigned long long error=0;
                for(std::size_t i=0;i<warm.size();i+=4){bool bad=false;for(unsigned c=0;c<4;++c){
                    unsigned delta=unsigned(std::abs(int(warm[i+c])-int(cold[i+c])));error+=delta;bad=bad || delta>2;}if(bad)++changed;}
                ok=changed<=warm.size()/4000 && error<=warm.size()/100;
                std::printf("UNIT post-draw terrain parity: %s changed=%zu error=%llu bytes=%zu\n",ok?"pass":"FAIL",changed,error,warm.size());
            }
        }
    }
#ifdef C3X_LAB_PREVIEW
    if(ok)ok=lab_compose_units(module,argv[5],frame.hour,tile_width,output);
#endif
    reset();
    FreeLibrary(module);
    return ok ? 0 : 1;
}

bool preview_units(HMODULE module,char const* path,int hour) {
    auto draw=reinterpret_cast<c3x_renderer_unit_draw_fn>(GetProcAddress(module,"c3x_renderer_unit_draw"));
    auto configure=reinterpret_cast<c3x_renderer_set_unit_rendering_fn>(GetProcAddress(module,"c3x_renderer_set_unit_rendering"));
    if(!draw || !configure)return false;
    BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth=1024;info.bmiHeader.biHeight=-1152;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
    HDC dc=CreateCompatibleDC(nullptr);void* bits=nullptr;
    HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
    if(!dc || !bitmap){if(bitmap)DeleteObject(bitmap);if(dc)DeleteDC(dc);return false;}
    auto old=SelectObject(dc,bitmap);bool ok=true;unsigned drawn=0;
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
                ++changes;if(x>=unit.body_x+extent || y>=unit.body_y+extent)ok=false;
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
                    ok=with_background(&unit,dc16,bg16)==C3X_RENDERER_RESULT_OK;GdiFlush();
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
            ok=with_background(&unit,dc,background)==C3X_RENDERER_RESULT_OK && ok;GdiFlush();
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
