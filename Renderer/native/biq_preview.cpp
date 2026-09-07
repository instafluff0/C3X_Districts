#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#include "c3x_renderer_api.h"

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

int main(int argc, char ** argv) {
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
    bool pickup=std::strcmp(profile,"pickup-r1")==0;
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
    if (set_definitions == nullptr || render == nullptr || reset == nullptr ||
        set_definitions(argv[2], argv[3], nullptr, nullptr) != C3X_RENDERER_RESULT_OK)
        return 1;

    char object_option[8]={};bool objects=GetEnvironmentVariableA("C3X_RENDERER_PREVIEW_OBJECTS",object_option,sizeof(object_option))!=0;
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
        tile.resource_id = tile.resource_class = tile.tile_building_id = -1;
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
        if(candidates.size()>=6){
            auto& city=tiles[candidates[0]];city.city_id=1;city.city_owner_id=1;city.city_size=2;
            city.city_culture_group=0;city.city_era=2;city.city_flags=C3X_RENDERER_CITY_CAPITAL|C3X_RENDERER_CITY_WALLED;
            auto& mine=tiles[candidates[1]];mine.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;mine.route_style=2;
            auto& farm=tiles[candidates[2]];farm.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;farm.irrigation_mask=15;farm.route_style=2;
            auto& resource=tiles[candidates[3]];resource.resource_id=1;resource.resource_class=0;strcpy_s(resource.resource_name,"Iron");
            auto& road=tiles[candidates[4]];road.road_mask=15;road.route_style=2;
            auto& rail=tiles[candidates[5]];rail.road_mask=15;rail.railroad_mask=15;rail.route_style=3;
        }
    }
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
    int result = render(&frame, &output);
    std::size_t expected_rendered = 0;
    for (c3x_renderer_tile_v1 const & tile : tiles)
        if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) != 0)
            ++expected_rendered;
    bool ok = result == C3X_RENDERER_RESULT_OK &&
              (pickup ? output.rendered_tile_count >= expected_rendered : output.rendered_tile_count == expected_rendered) &&
              output.fallback_tile_count == 0 && write_bmp(argv[5], output);
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
            ok=render(&frame,&output)==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 &&
                output.geometry_tiles_built>0 && output.geometry_tiles_reused>0;
            std::printf("PICKUP authoritative edit: %s active=%u built=%u reused=%u\n",ok?"pass":"FAIL",
                unsigned(active),output.geometry_tiles_built,output.geometry_tiles_reused);
            std::vector<unsigned char> edited;
            if(ok){auto first=static_cast<unsigned char const*>(output.bgra_pixels);
                edited.assign(first,first+output.stride_bytes*output.height);}
            reset();
            if(set_definitions(argv[2],argv[3],nullptr,nullptr)!=C3X_RENDERER_RESULT_OK ||
                render(&frame,&output)!=C3X_RENDERER_RESULT_OK)ok=false;
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
            ok=render(&frame,&output)==C3X_RENDERER_RESULT_OK && ok;QueryPerformanceCounter(&b);
            times.push_back(double(b.QuadPart-a.QuadPart)*1000/frequency.QuadPart);
            if(output.geometry_tiles_built || output.geometry_upload_bytes)ok=false;
        }
        std::sort(times.begin(),times.end());
        std::printf("PICKUP selection p95_ms=%.3f max_ms=%.3f\n",times[30],times[31]);
        auto prepare=[&](){
            LARGE_INTEGER begin={},now={};QueryPerformanceCounter(&begin);
            for(;;){
                Sleep(25);int code=render(&frame,&output);QueryPerformanceCounter(&now);
                if(code!=C3X_RENDERER_RESULT_OK)return false;
                if(!output.prefetch_tiles_pending && !output.prefetch_blocks_pending)break;
                if(now.QuadPart-begin.QuadPart>frequency.QuadPart*60)break;
            }
            std::printf("PICKUP prepared pending=%u tiles=%u unavailable=%u blocks=%u bytes=%u\n",
                output.prefetch_tiles_pending,output.prefetch_tiles_built,output.prefetch_tiles_unavailable,
                output.prefetch_blocks_built,output.prefetch_cache_bytes);return true;
        };
        ok=prepare() && ok;
        int original_x=center_x,original_y=center_y;
        int steps[][2]={{4,0},{8,0},{8,4},{4,4},{0,0}};
        for(auto const& step:steps){
            center_x=original_x+step[0];center_y=original_y+step[1];tiles=capture_view();
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            LARGE_INTEGER a={},b={};QueryPerformanceCounter(&a);
            int code=render(&frame,&output);QueryPerformanceCounter(&b);
            ok=ok && code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;
            std::printf("PICKUP jump=%d,%d result=%d ms=%.3f built=%u reused=%u gpu_bytes=%u geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f\n",
                step[0],step[1],code,double(b.QuadPart-a.QuadPart)*1000/frequency.QuadPart,
                output.geometry_tiles_built,output.geometry_tiles_reused,output.geometry_cache_bytes,
                double(output.geometry_ticks)*1000/frequency.QuadPart,double(output.draw_ticks)*1000/frequency.QuadPart,
                double(output.readback_ticks)*1000/frequency.QuadPart);
            if(code!=C3X_RENDERER_RESULT_OK)break;
            ok=prepare() && ok;
        }
        // Compare an image assembled from prepared pixel blocks with a fresh
        // renderer of the same authoritative snapshot. Use the existing native
        // pixel-budget thresholds, never a looser pickup-only allowance.
        center_x=original_x+4;center_y=original_y+4;tiles=capture_view();
        frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        if(render(&frame,&output)!=C3X_RENDERER_RESULT_OK)ok=false;
        std::vector<unsigned char> warm;
        if(ok){auto first=static_cast<unsigned char const*>(output.bgra_pixels);
            warm.assign(first,first+output.stride_bytes*output.height);
            std::string path=std::string(argv[5])+".cached.bmp";write_bmp(path.c_str(),output);}
        reset();
        if(set_definitions(argv[2],argv[3],nullptr,nullptr)!=C3X_RENDERER_RESULT_OK ||
            render(&frame,&output)!=C3X_RENDERER_RESULT_OK)ok=false;
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
    reset();
    FreeLibrary(module);
    return ok ? 0 : 1;
}
