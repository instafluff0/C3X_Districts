"""Execute current native city/overlay authority and copied world-reader bodies.

This host C harness extracts the injected helpers plus Civ III's native city
eligibility and remembered-overlay methods. It cannot dispatch a Windows build.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from Renderer.native.test_injected_world_local import production_function


ROOT = Path(__file__).resolve().parents[2]


def native_body(source, name):
    function = source.split(name, 1)[1].split("\n}\n", 1)[0]
    return function[function.index("{\n") + 2:]


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "Renderer/native/c3x_renderer_api.h"
#define __ 0
/*PRODUCTION_SQUARE_TYPES*/
typedef struct c3x_renderer_tile_v1 c3x_renderer_tile_v1;
typedef unsigned uint;
typedef unsigned char byte;
typedef unsigned short ushort;
typedef struct Tile Tile;
typedef struct City {
    struct {int ID,X,Y,CivID;struct {int Size;}Population;}Body;
} City;
typedef struct Map_Renderer {int identity;} Map_Renderer;
typedef struct TileVtable {
    bool (*m17_Check_Irrigation)(Tile*,int,int);
    int (*m18_Check_Mines)(Tile*,int,int);
    int (*m20_Check_Pollution)(Tile*,int,int);
    int (*m21_Check_Crates)(Tile*,int,int);
    int (*m15_Check_Goody_Hut)(Tile*,int,int);
    int (*m7_Check_Barbarian_Camp)(Tile*,int,int);
    int (*m44_Get_Barbarian_TribeID)(Tile*);
    int (*m45_Get_City_ID)(Tile*);
    int (*m25_Check_Roads)(Tile*,int,int);
    int (*m23_Check_Railroads)(Tile*,int,int);
} TileVtable;
struct Tile {
    TileVtable* vtable;
    struct {unsigned Fog_Of_War,FOWStatus,V3,Visibility,field_D0_Visibility;
            unsigned char Visibile_Overlays[32];}Body;
    unsigned Overlays;
    int city_id,tribe,live_resource,live_building,live_effect;
};
typedef struct Map Map;
typedef struct MapVtable {char (*m10_Get_Map_Zoom)(Map*);} MapVtable;
struct Map {MapVtable* vtable;int Width,Height,Flags,Seed;Map_Renderer Renderer;};
typedef struct Leader {int ID,RaceID,Era,CapitalID;} Leader;
typedef struct Race {int CultureGroupID;char CountryName[64],SingularName[64];} Race;
typedef struct Era {struct {char S[64];}Name;} Era;
struct Improvement {int Combat_Bombard;};
typedef struct Bic {Map Map;bool is_zoomed_out;
    struct {int MaximumSize_City,MaximumSize_Town;}General;
    int RacesCount,ErasCount,ImprovementsCount;
    Race* Races;Era* Eras;struct Improvement* Improvements;
} Bic;
static Bic bic_data,*p_bic_data=&bic_data;
static Leader leaders[32];static Race races[2];static Era eras[4];
static struct Improvement improvements[1];
static Tile tiles[32],null_tile,*p_null_tile=&null_tile;
static City city;
static unsigned topology[32],debug_mode_bits;
static struct {unsigned* custom_renderer_world_topology;}state,*is=&state;
static bool online,visible,known,wall;
static int full_reads,city_reads,tribe_reads,wall_reads,native_draws;
static int last_viewer;
static char map_zoom(Map* value){assert(value==&bic_data.Map);return 0;}
static MapVtable map_vtable={map_zoom};
static Tile* tile_at(int x,int y){
    if(x<0||y<0||x>=bic_data.Map.Width||y>=bic_data.Map.Height||((x+y)&1))return p_null_tile;
    return &tiles[(y*bic_data.Map.Width+x)/2];
}
static Tile* native_get_tile(Map* value,unsigned index){assert(value==&bic_data.Map&&index<32);return &tiles[index];}
static bool is_online_game(void){return online;}
static City* city_at(int x,int y){Tile* tile=tile_at(x,y);return tile!=p_null_tile&&tile->city_id==city.Body.ID?&city:NULL;}
static City* get_city_ptr(int id){return id==city.Body.ID?&city:NULL;}
static void native_draw_city(City* value,Map_Renderer* renderer,int x,int y,int a,int b,int c,int scale){
    assert(value==&city&&renderer==&bic_data.Map.Renderer&&x==64&&y==32);
    assert(a==0&&b==1&&c==1&&scale==1);++native_draws;
}
static bool has_active_building(City* value,int id){assert(value==&city&&id==0);++wall_reads;return wall;}
static int clamp(int low,int high,int value){return value<low?low:value>high?high:value;}
static void wrap_tile_coords(Map* value,int* x,int* y){
    if(value->Flags&1)*x=(*x%value->Width+value->Width)%value->Width;
    if(value->Flags&2)*y=(*y%value->Height+value->Height)%value->Height;
}
static unsigned capture_custom_renderer_visibility(Tile* tile,int viewer,int x,int y){
    assert(tile==tile_at(x,y));if(!known)return 0;
    unsigned result=C3X_RENDERER_TILE_VISIBILITY_KNOWN;
    if(viewer==0||((debug_mode_bits&8)&&!online))return result|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
    if(tile->Body.Fog_Of_War&(1u<<leaders[viewer].ID))result|=C3X_RENDERER_TILE_EXPLORED;
    if(visible)result|=C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
    return result;
}
static bool read_custom_renderer_tile(c3x_renderer_tile_v1* record,int viewer,int px,int py,
    int mask,int tile_x,int tile_y,Tile* tile,bool topology_only,bool capture_units){
    assert(viewer==1&&px==0&&py==0&&mask==77&&tile_x==2&&tile_y==2&&tile==tile_at(tile_x,tile_y));
    assert(!topology_only&&!capture_units);++full_reads;
    *record=(c3x_renderer_tile_v1){0};record->tile_flags=capture_custom_renderer_visibility(tile,viewer,tile_x,tile_y)|C3X_RENDERER_TILE_RENDER;
    unsigned ground=topology[(tile_y*bic_data.Map.Width+tile_x)/2];
    record->tile_x=tile_x;record->tile_y=tile_y;
    record->terrain_type=ground&255u;record->real_terrain_type=(ground>>8)&255u;
    record->river_code=(ground>>16)&255u;
    /*PRODUCTION_FOREGROUND_SEED*/
    /*PRODUCTION_FOREGROUND_FEATURES*/
    record->resource_id=tile->live_resource;record->tile_building_id=tile->live_building;record->has_effect=tile->live_effect;
    return true;
}
'''


QUERIES = r'''
static int overlay_bit(Tile* tile,int unused,int viewer,int bit){
    assert(unused==0);last_viewer=viewer;return (int)(native_overlays(tile,(byte)viewer)&(1u<<bit));
}
static bool irrigation(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,3)!=0;}
static int mine(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,2)!=0;}
static int pollution(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,6)!=0;}
static int crater(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,4)!=0;}
static int hut(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,5)!=0;}
static int camp(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,7)!=0;}
static int roads(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,0)!=0;}
static int rail(Tile* tile,int unused,int viewer){return overlay_bit(tile,unused,viewer,1)!=0;}
static int tribe(Tile* tile){++tribe_reads;return tile->tribe;}
static int city_id(Tile* tile){++city_reads;return tile->city_id;}
static TileVtable tile_vtable={irrigation,mine,pollution,crater,hut,camp,tribe,city_id,roads,rail};
static void reset(void){
    memset(tiles,0,sizeof tiles);memset(topology,0,sizeof topology);
    memset(&bic_data,0,sizeof bic_data);memset(leaders,0,sizeof leaders);
    memset(races,0,sizeof races);memset(eras,0,sizeof eras);
    bic_data.Map=(Map){&map_vtable,8,8,0,0x13579bdf,{1}};
    bic_data.General.MaximumSize_Town=6;bic_data.General.MaximumSize_City=12;
    bic_data.RacesCount=2;bic_data.ErasCount=4;bic_data.ImprovementsCount=1;
    bic_data.Races=races;bic_data.Eras=eras;bic_data.Improvements=improvements;
    improvements[0].Combat_Bombard=1;
    leaders[1]=(Leader){1,0,1,99};leaders[2]=(Leader){2,1,3,4};
    races[0].CultureGroupID=2;races[1].CultureGroupID=4;
    strcpy(races[0].CountryName,"First");strcpy(races[1].CountryName,"Second");
    strcpy(eras[1].Name.S,"Old");strcpy(eras[3].Name.S,"New");
    city=(City){{4,2,2,1,{3}}};
    for(int n=0;n<32;++n){tiles[n].vtable=&tile_vtable;tiles[n].city_id=-1;topology[n]=2u|(7u<<8)|(9u<<16);}
    Tile* tile=tile_at(2,2);tile->city_id=4;tile->Body.Fog_Of_War=2;
    tile->live_resource=91;tile->live_building=92;tile->live_effect=1;
    state.custom_renderer_world_topology=topology;
    known=true;visible=online=wall=false;debug_mode_bits=0;
    full_reads=city_reads=tribe_reads=wall_reads=native_draws=0;last_viewer=-1;
}
static c3x_renderer_tile_v1 read_record(void){c3x_renderer_tile_v1 result;
    assert(read_custom_renderer_world_record(&result,1,77,2,2,tile_at(2,2)));return result;
}
static void omitted(c3x_renderer_tile_v1 const* record){
    assert(!(record->tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)));
    assert(record->resource_id==-1&&record->resource_class==-1&&record->tile_building_id==-1&&!record->has_effect);
    assert(record->unit_type_id==-1&&record->unit_owner_id==-1&&record->unit_class==-1&&record->unit_state==-1);
    assert(record->unit_damage==-1&&record->unit_direction==-1&&!record->unit_type_name[0]);
    assert(!record->resource_name[0]&&record->territory_owner_id==-1);
    assert(full_reads==0);
}
static void hidden_city(void){
    reset();native_city(1,2,2,&bic_data.Map.Renderer,0,0);assert(native_draws==1);
    c3x_renderer_tile_v1 old=read_record();omitted(&old);
    assert(old.city_id==4&&old.city_size==0&&old.city_owner_id==1&&old.city_culture_group==2&&old.city_era==1);
    assert((old.tile_flags&(C3X_RENDERER_TILE_CITY_BODY_KNOWN|C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN))==
       (C3X_RENDERER_TILE_CITY_BODY_KNOWN|C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN));
    city.Body.Population.Size=13;city.Body.CivID=2;
    c3x_renderer_tile_v1 next=read_record();omitted(&next);
    assert(next.city_size==2&&next.city_owner_id==2&&next.city_culture_group==4&&next.city_era==3);
    assert(next.city_flags&C3X_RENDERER_CITY_CAPITAL);assert(!strcmp(next.city_owner,"Second"));
    city.Body.Population.Size=3;city.Body.CivID=1;wall=true;next=read_record();
    assert(next.city_flags&C3X_RENDERER_CITY_WALLED);
    tile_at(2,2)->city_id=-1;native_draws=0;native_city(1,2,2,&bic_data.Map.Renderer,0,0);assert(!native_draws);
    next=read_record();assert(next.city_id==-1&&next.city_owner_id==-1&&next.city_size==-1&&next.city_era==-1);
    assert(!next.city_population&&!next.city_flags&&!next.city_owner[0]);
    assert(next.tile_flags&C3X_RENDERER_TILE_CITY_BODY_KNOWN);
}
static void unexplored(void){
    reset();tile_at(2,2)->Body.Fog_Of_War=0;
    native_city(1,2,2,&bic_data.Map.Renderer,0,0);assert(!native_draws);
    c3x_renderer_tile_v1 record=read_record();omitted(&record);
    assert(record.terrain_type==2&&record.real_terrain_type==7&&record.river_code==9);
    assert(!(record.tile_flags&(C3X_RENDERER_TILE_CITY_BODY_KNOWN|C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN)));
    assert(!city_reads&&!wall_reads&&!tribe_reads&&record.city_id==-1);
    known=false;assert(!read_custom_renderer_world_record(&record,1,77,2,2,tile_at(2,2)));
}
static void remembered(void){
    reset();Tile* tile=tile_at(2,2);
    tile->Overlays=0;tile->Body.Visibile_Overlays[1]=0x3f;
    tile_at(1,1)->Body.Visibile_Overlays[1]=8;tile_at(3,3)->Body.Visibile_Overlays[1]=8;
    c3x_renderer_tile_v1 record=read_record();omitted(&record);
    assert(record.road_mask==1&&record.railroad_mask==1&&record.irrigation_mask==9&&record.route_style==1);
    assert((record.improvement_flags&(C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION|C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_CRATER))==
       (C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION|C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_CRATER));
    tile->Overlays=0xff;tile->Body.Visibile_Overlays[1]=0;record=read_record();
    assert(!record.road_mask&&!record.railroad_mask&&!record.irrigation_mask&&!record.improvement_flags&&!tribe_reads);
    // Overlay removals replace only their admitted subset when used on an
    // existing body; the tile-building and other record fields survive.
    record.improvement_flags=127;record.tile_building_id=12;record.resource_id=13;record.has_effect=1;
    capture_custom_renderer_overlay_body(&record,tile,1,2,2);
    assert(record.improvement_flags==C3X_RENDERER_IMPROVEMENT_TILE_BUILDING);
    assert(record.tile_building_id==12&&record.resource_id==13&&record.has_effect==1);
    // Native m42 changes to current overlays only when that viewer is visible.
    tile->Body.Visibility=2;record=read_record();assert(record.road_mask==1&&record.railroad_mask==1);
    assert(record.improvement_flags&C3X_RENDERER_IMPROVEMENT_POLLUTION);
}
static void camp_admission(void){
    reset();Tile* tile=tile_at(2,2);tile->tribe=5;
    tile->Body.Visibile_Overlays[1]=0x80;tile->Overlays=0;
    c3x_renderer_tile_v1 record=read_record();assert(!tribe_reads&&record.barbarian_tribe_id==-1);
    assert(!(record.improvement_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP));
    tile->Overlays=0x80;record=read_record();assert(tribe_reads==1&&record.barbarian_tribe_id==5);
    tile->Body.Visibile_Overlays[1]=0;record=read_record();assert(tribe_reads==1&&record.barbarian_tribe_id==-1);
    tile->Body.Visibility=2;record=read_record();assert(tribe_reads==2&&record.barbarian_tribe_id==5);
}
static void full_visible(void){
    reset();visible=true;c3x_renderer_tile_v1 record=read_record();
    assert(full_reads==1&&(record.tile_flags&C3X_RENDERER_TILE_PREFETCH));
    assert(record.tile_flags&C3X_RENDERER_TILE_VISIBLE);assert(!(record.tile_flags&C3X_RENDERER_TILE_RENDER));
    assert(record.resource_id==91&&record.tile_building_id==92&&record.has_effect);
}
static void deterministic_seed(void){
    int const seeds[]={0,0x13579bdf,-1};
    unsigned const expected[]={0x0a836984u,0x19d4f25bu,0xf57c967bu};
    for(unsigned i=0;i<sizeof seeds/sizeof *seeds;++i){
        reset();bic_data.Map.Seed=seeds[i];
        c3x_renderer_tile_v1 record=read_record(),again=read_record();omitted(&record);
        assert(record.variant_seed==expected[i]&&again.variant_seed==record.variant_seed);
        Tile* neighbor=tile_at(3,3);neighbor->Body.Fog_Of_War=2;
        c3x_renderer_tile_v1 moved;
        assert(read_custom_renderer_world_record(&moved,1,77,3,3,neighbor));omitted(&moved);
        assert(moved.variant_seed!=record.variant_seed);
        visible=true;c3x_renderer_tile_v1 foreground=read_record();
        assert(foreground.variant_seed==record.variant_seed);
    }
}
static void terrain_recipe(void){
    int const terrains[]={SQ_Forest,SQ_Jungle,SQ_Swamp,SQ_Volcano,SQ_Grassland};
    unsigned const features[]={C3X_RENDERER_FEATURE_FOREST,C3X_RENDERER_FEATURE_JUNGLE,
        C3X_RENDERER_FEATURE_MARSH,C3X_RENDERER_FEATURE_VOLCANO,0};
    for(unsigned i=0;i<sizeof terrains/sizeof *terrains;++i){
        reset();topology[(2*bic_data.Map.Width+2)/2]=SQ_Grassland|((unsigned)terrains[i]<<8)|(9u<<16);
        c3x_renderer_tile_v1 fog=read_record();omitted(&fog);
        assert((fog.tile_flags&(C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED))==
            (C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED));
        assert(!(fog.tile_flags&C3X_RENDERER_TILE_VISIBLE));
        assert(fog.terrain_type==SQ_Grassland&&fog.real_terrain_type==terrains[i]&&fog.feature_flags==features[i]);
        visible=true;c3x_renderer_tile_v1 foreground=read_record();
        assert(foreground.terrain_type==fog.terrain_type&&foreground.real_terrain_type==fog.real_terrain_type);
        assert(foreground.feature_flags==fog.feature_flags&&foreground.variant_seed==fog.variant_seed);
        assert(foreground.river_code==fog.river_code);
    }
}
static void hidden_objects(void){
    reset();c3x_renderer_tile_v1 before=read_record();omitted(&before);
    Tile* tile=tile_at(2,2);tile->live_resource=123;tile->live_building=456;tile->live_effect=0;
    c3x_renderer_tile_v1 after=read_record();omitted(&after);
    assert(!memcmp(&before,&after,sizeof before));
    visible=true;after=read_record();
    assert(full_reads==1&&after.resource_id==123&&after.tile_building_id==456&&!after.has_effect);
}
int main(int argc,char** argv){
    assert(argc==2);
    if(!strcmp(argv[1],"city"))hidden_city();
    else if(!strcmp(argv[1],"unexplored"))unexplored();
    else if(!strcmp(argv[1],"remembered"))remembered();
    else if(!strcmp(argv[1],"camp"))camp_admission();
    else if(!strcmp(argv[1],"visible"))full_visible();
    else if(!strcmp(argv[1],"seed"))deterministic_seed();
    else if(!strcmp(argv[1],"terrain"))terrain_recipe();
    else if(!strcmp(argv[1],"hidden"))hidden_objects();
    else assert(false);
}
'''


class InjectedWorldAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("Host clang unavailable")
        temporary = tempfile.TemporaryDirectory(prefix="c3x-world-authority-")
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        injected = (ROOT / "injected_code.c").read_text()
        native = (ROOT / "ref/Civ3Conquests_master.exe.c").read_text()
        header = (ROOT / "Civ3Conquests.h").read_text()
        square_types = "enum SquareTypes" + header.split("enum SquareTypes", 1)[1].split("};", 1)[0] + "};"
        foreground = production_function(injected, "read_custom_renderer_tile").splitlines()
        seed = [line for line in foreground if "record->variant_seed =" in line]
        features = [line for line in foreground if "record->real_terrain_type ==" in line and "record->feature_flags |=" in line]
        if len(seed) != 1 or len(features) != 4:
            raise AssertionError("Current foreground terrain recipe extraction changed")
        # Use the real foreground expressions rather than duplicating their
        # seed formula or feature mapping in the authored full-reader mock.
        prelude = PRELUDE.replace("/*PRODUCTION_SQUARE_TYPES*/", square_types)
        prelude = prelude.replace("/*PRODUCTION_FOREGROUND_SEED*/", seed[0])
        prelude = prelude.replace("/*PRODUCTION_FOREGROUND_FEATURES*/", "\n".join(features))
        overlays = native_body(native, "Tile::impl_m42_Get_Overlays")
        overlays = overlays.replace("this", "tile")
        city = native_body(native, "Map_Renderer::impl_m25_Draw_City_Image")
        city = city.replace("Map::get_tile", "native_get_tile").replace("City::draw_on_map", "native_draw_city")
        program = prelude + "\nstatic unsigned native_overlays(Tile* tile,byte visible_to_civ){\n" + overlays + "\n}\n"
        program += "static void native_city(int param_1,int tile_x,int tile_y,Map_Renderer* map_renderer,int pixel_x,int pixel_y){\n" + city + "\n}\n"
        for name in ("capture_custom_renderer_city_body", "capture_custom_renderer_overlay_body", "read_custom_renderer_world_record"):
            program += production_function(injected, name)
        # Native query definitions are needed by the extracted helper only at
        # runtime through its vtable; no authored replacement helper is used.
        program += QUERIES
        path = directory / "contract.c"
        path.write_text(program)
        cls.executable = directory / "contract"
        result = subprocess.run([compiler, "-std=c17", "-O1", "-Wall", "-Wextra", "-Werror",
                                 "-I", str(ROOT), str(path), "-o", str(cls.executable)],
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def contract(self, name):
        result = subprocess.run([str(self.executable), name], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explored_city_updates_current_recipe_and_explicit_absence(self):
        self.contract("city")

    def test_unexplored_and_unknown_tiles_cannot_authorize_partial_art(self):
        self.contract("unexplored")

    def test_native_remembered_overlays_and_removal_use_viewer_authority(self):
        self.contract("remembered")

    def test_camp_requires_native_live_presence_then_viewer_admission(self):
        self.contract("camp")

    def test_visible_world_copy_preserves_full_route_and_disables_units(self):
        self.contract("visible")

    def test_explored_fog_seed_is_repeatable_and_matches_foreground(self):
        self.contract("seed")

    def test_explored_fog_terrain_features_match_foreground_recipe(self):
        self.contract("terrain")

    def test_hidden_optional_objects_do_not_change_explored_fog_record(self):
        self.contract("hidden")


if __name__ == "__main__":
    unittest.main()
