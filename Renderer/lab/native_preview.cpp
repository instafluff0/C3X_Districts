// Standalone category witness using the actual production terrain and unit APIs.
// Reuse the existing capture/ownership harness; never launch or patch the game.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vector>
#include "../native/c3x_renderer_api.h"
void lab_place_objects(std::vector<c3x_renderer_tile_v1>& tiles, int center_x, int center_y, int map_width);
bool lab_verify_objects(c3x_renderer_frame_v1 const& frame, c3x_renderer_output_v1 const& output);
bool lab_compose_units(HMODULE module, char const* image_path, int hour, int tile_width,
                       c3x_renderer_output_v1 const& terrain);
#define C3X_LAB_PREVIEW 1
#define main terrain_preview_main
#include "../native/biq_preview.cpp"
#undef main

void lab_place_objects(std::vector<c3x_renderer_tile_v1>& tiles, int center_x, int center_y, int map_width) {
    char category[32]={}, water[8]={};
    GetEnvironmentVariableA("C3X_LAB_OBJECT_STUDY",category,sizeof(category));
    GetEnvironmentVariableA("C3X_LAB_WATER_STUDY",water,sizeof(water));
    bool resources=std::strcmp(category,"resources")==0;
    bool infrastructure=std::strcmp(category,"infrastructure")==0;
    bool shadows=std::strcmp(category,"shadows")==0;
    if(!resources && !infrastructure && !shadows)return;
    char const*land[]={"Iron","Cattle","Horses","Wheat","Gold","Dyes"};
    char const*sea[]={"Fish","Whales"};
    for(auto& tile:tiles) {
        int x=((tile.tile_x%map_width)+map_width)%map_width-center_x, y=tile.tile_y-center_y;
        if(shadows) {
            if(x==-4 && y==2) {
                tile.city_id=1;tile.city_owner_id=1;tile.city_size=1;
                tile.city_culture_group=0;tile.city_era=0;tile.city_flags=0;
            }
            if(y==0 && (x==0 || x==4)) {
                tile.resource_id=x==0?100:101;tile.resource_class=0;
                strcpy_s(tile.resource_name,x==0?"Iron":"Horses");
            }
        } else if(resources) {
            unsigned count=water[0]?2u:6u;
            for(unsigned i=0;i<count;++i) {
                int sx=water[0]?(int(i)*4-2):(int(i%3)*2-2);
                int sy=water[0]?0:(int(i/3)*4-2);
                if(x==sx && y==sy){
                    tile.resource_id=100+int(i);tile.resource_class=0;
                    strcpy_s(tile.resource_name,water[0]?sea[i]:land[i]);
                }
            }
        } else {
            // Actual connected nodes: two horizontal runs joined by a branch.
            if((std::abs(y)==2 && x>=-6 && x<=6) || (x==0 && y>=-6 && y<=6)) {
                tile.road_mask=15;tile.route_style=2;
                if(y==2)tile.railroad_mask=15;
            }
            if(x==-2 && y==-4)tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;
            if(x==2 && y==-4){tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;tile.irrigation_mask=15;}
        }
    }
}

bool lab_verify_objects(c3x_renderer_frame_v1 const& frame, c3x_renderer_output_v1 const& output) {
    char category[32]={},water[8]={};
    GetEnvironmentVariableA("C3X_LAB_OBJECT_STUDY",category,sizeof(category));
    GetEnvironmentVariableA("C3X_LAB_WATER_STUDY",water,sizeof(water));
    bool resources=std::strcmp(category,"resources")==0;
    bool infrastructure=std::strcmp(category,"infrastructure")==0;
    bool shadows=std::strcmp(category,"shadows")==0;
    if(!resources && !infrastructure && !shadows)return true;
    bool ok=output.replacement_tile_count==frame.tile_count && output.replacement_tile_flags;
    unsigned count=0,ownership=0;
    for(unsigned i=0;i<frame.tile_count && ok;++i) {
        auto const& tile=frame.tiles[i];
        if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
        auto flags=output.replacement_tile_flags[i];ownership|=flags;
        if((resources || shadows) && tile.resource_id>=0){
            ++count;ok=(flags&C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED)!=0;
            if(!ok)std::printf("FAIL missing custom resource: %s\n",tile.resource_name);
        }
    }
    if(resources)ok=ok && count==(water[0]?2u:6u);
    if(shadows)ok=ok && count==2 && (ownership&C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED)!=0;
    if(infrastructure){
        unsigned expected=C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED|C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED|
            C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED|C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED;
        ok=ok && (ownership&expected)==expected;
    }
    std::printf("%s category object study: %s resources=%u ownership=%u\n",ok?"PASS":"FAIL",category,count,ownership);
    return ok;
}

bool lab_compose_units(HMODULE module, char const* image_path, int hour, int tile_width,
                       c3x_renderer_output_v1 const& terrain) {
    char enabled[32] = {};
    if (!GetEnvironmentVariableA("C3X_LAB_UNIT_STUDY", enabled, sizeof(enabled))) return true;
    auto draw = reinterpret_cast<c3x_renderer_unit_draw_background_fn>(GetProcAddress(module, "c3x_renderer_unit_draw_background"));
    bool ok = draw != nullptr;
    HDC dc = CreateCompatibleDC(nullptr);
    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = terrain.width;
    info.bmiHeader.biHeight = -terrain.height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    void* pixels = nullptr;
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &pixels, nullptr, 0);
    ok = bitmap && dc && pixels && terrain.bgra_pixels && ok;
    if (ok) for (int row=0; row<terrain.height; ++row)
        std::memcpy(static_cast<unsigned char*>(pixels)+std::size_t(row)*terrain.width*4,
                    static_cast<unsigned char const*>(terrain.bgra_pixels)+std::size_t(row)*terrain.stride_bytes,
                    std::size_t(terrain.width)*4);
    HGDIOBJ previous = bitmap && dc ? SelectObject(dc, bitmap) : nullptr;
    char cursor[16] = {};
    GetEnvironmentVariableA("C3X_LAB_ACTION_CURSOR", cursor, sizeof(cursor));
    char const* keys[] = {"Warrior", "Settler", "Worker", "Horseman", "Tank", "Fighter"};
    char const* study_keys[] = {"warrior", "spearman", "pikeman", "archer", "settler", "worker"};
    char const* variants[] = {"baseline", "anatomy", "ssaa"};
    char const* labels[] = {"Current fit / 1x material sampling", "Anatomy fit / 1x material sampling", "Anatomy fit / 2x material sampling"};
    bool sizing=std::strncmp(enabled,"sizing",6)==0;
    if(sizing && std::strcmp(enabled,"sizing-gameplay")!=0 && pixels)
        std::fill_n(static_cast<unsigned*>(pixels),std::size_t(terrain.width)*terrain.height,0xff929c9bu);
    bool shadows=std::strcmp(enabled,"shadows")==0;
    int count=sizing?18:shadows?2:6;
    if(sizing && dc) {SetBkMode(dc,TRANSPARENT);SetTextColor(dc,RGB(245,245,245));}
    for (int index = 0; index < count && ok; ++index) {
        c3x_renderer_unit_v1 unit = {};
        unit.struct_size = sizeof(unit);
        unit.unit_id = index;
        sprintf_s(unit.unit_key, "PRTO_%s", keys[index%6]);
        unit.action = 2;
        unit.action_cursor = cursor[0] ? std::atoi(cursor) : 7;
        unit.frame_count = 16;
        unit.direction = 3;
        unit.hour = hour;
        unit.sprite_width = unit.sprite_height = 191;
        unit.reduced = tile_width == 64;
        unit.projection_scale_milli=tile_width*1000/128;
        unit.display_color_rgb = 0x205bdd;
        unit.body_x = 150 + (index % 3) * 160 - 191*unit.projection_scale_milli/2000;
        unit.body_y = 215 + (index / 3) * 150 - 191*unit.projection_scale_milli/2000;
        if(sizing) {
            int column=index%6,row=index/6;
            sprintf_s(unit.unit_key,"PRTO_Lab_%s_%s",variants[row],study_keys[column]);
            unit.action=std::strcmp(enabled,"sizing-move")==0?2:1;
            unit.action_cursor=unit.action==1?0:7;
            unit.presentation_frequency=1000;unit.presentation_time_ticks=0;
            unit.sprite_width=unit.sprite_height=320;
            unit.projection_scale_milli=tile_width*1000/128;
            int half=unit.sprite_width*unit.projection_scale_milli/2000;
            unit.body_x=100+column*200-half;
            unit.body_y=265+row*290-half;
            if(column==0)TextOutA(dc,12,12+row*290,labels[row],int(std::strlen(labels[row])));
            TextOutA(dc,70+column*200,274+row*290,study_keys[column],int(std::strlen(study_keys[column])));
        }
        if(shadows) {
            // Flat receiving tiles south of the mixed static/animated scene.
            unit.body_x=terrain.width/2+(index*2)*tile_width/2-191/(unit.reduced?4:2);
            unit.body_y=terrain.height/2+3*tile_width/4-191/(unit.reduced?4:2);
        }
        ok = draw(&unit, dc, dc) == C3X_RENDERER_RESULT_OK;
        if (!ok) std::printf("FAIL category unit %s\n", unit.unit_key);
    }
    GdiFlush();
    if (ok) {
        c3x_renderer_output_v1 output = {};
        output.width = terrain.width;
        output.height = terrain.height;
        output.stride_bytes = terrain.width*4;
        output.bgra_pixels = pixels;
        ok = write_bmp(image_path, output);
    }
    if (previous) SelectObject(dc, previous);
    if (bitmap) DeleteObject(bitmap);
    if (dc) DeleteDC(dc);
    std::printf("%s category unit study: %d current production families, native body API\n", ok ? "PASS" : "FAIL",count);
    return ok;
}

int main(int argc, char** argv) {
    // Publish this invocation's real process handle before any expensive
    // shader compilation. A transport timeout is not a native process exit.
    char path[1024]={},id[64]={};
    if(GetEnvironmentVariableA("C3X_LAB_PID_FILE",path,sizeof(path)) &&
       GetEnvironmentVariableA("C3X_LAB_RUN_ID",id,sizeof(id))) {
        FILE* file=nullptr;
        if(fopen_s(&file,path,"w") || !file)return 1;
        std::fprintf(file,"%s %lu\n",id,GetCurrentProcessId());std::fclose(file);
    }
    return terrain_preview_main(argc, argv);
}
