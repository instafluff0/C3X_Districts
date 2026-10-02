"""Execute the configured sight gate and accepted move callback with host C mocks."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from Renderer.native.test_injected_world_local import production_function

ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = (
    "custom_renderer_tile_near_view",
    "custom_renderer_sight_near_view",
    "refresh_custom_renderer_sight",
    "notify_custom_renderer_unit_move",
)

PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include "Renderer/native/c3x_renderer_api.h"
#define __ 0
#define __cdecl
#define SBORROW4(a,b) (((long long)(a)-(b)<INT_MIN)||((long long)(a)-(b)>INT_MAX))
typedef struct LARGE_INTEGER { long long QuadPart; } LARGE_INTEGER;
typedef struct Map { int Width, Height, Flags; } Map;
typedef struct Unit { struct { int ID, X, Y, CivID;
    struct { struct { int current_anim_type; } summary; } Animation;
} Body; } Unit;
typedef struct Main_Screen_Form {
    int TileX_Min, TileX_Max, TileY_Min, TileY_Max;
    Unit *Current_Unit;
    struct { int *field_18E4; } animator;
} Main_Screen_Form;
typedef struct State {
    struct { bool enable_custom_rendering; } current_config;
    int custom_renderer_viewer_civ_id;
    void *custom_renderer_city_site_grades;
    unsigned custom_renderer_city_site_grade_count, custom_renderer_dirty_flags;
    bool custom_renderer_redraw_pending, custom_renderer_world_audit_needed;
    long long custom_renderer_map_epoch, custom_renderer_viewer_epoch;
    LARGE_INTEGER custom_renderer_qpc_frequency;
    long long (*custom_renderer_visual_clock)(void);
    c3x_renderer_world_move_fn custom_renderer_world_move;
    c3x_renderer_world_move_sight_fn custom_renderer_world_move_sight;
    c3x_renderer_world_reconcile_fn custom_renderer_world_reconcile;
    c3x_renderer_unit_move_fn custom_renderer_unit_move;
} State;
static State state, *is = &state;
static struct { Map Map; } bic, *p_bic_data = &bic;
static Main_Screen_Form screen, *p_main_screen_form = &screen;
static int animator_words[22], rings, configured_calls, legacy_calls, reconciles, movement_calls, states;
static int last_rings, world_result;
static bool target_visible;
static struct c3x_renderer_unit_move_v1 last_move;
int calc_max_visibility_range(void) { return rings; }
bool Map_in_range(Map *map,int unused,int x,int y) {
    (void)unused;return x>=0&&y>=0&&x<map->Width&&y<map->Height;
}
void wrap_tile_coords(Map *map,int *x,int *y) {
    if(map->Flags&1)*x=(*x%map->Width+map->Width)%map->Width;
    if(map->Flags&2)*y=(*y%map->Height+map->Height)%map->Height;
}
bool custom_renderer_tile_visible_at(int x,int y) {
    assert(Map_in_range(&bic.Map,0,x,y));return target_visible;
}
bool QueryPerformanceCounter(LARGE_INTEGER *now) { now->QuadPart=123456;return true; }
long long clock_ticks(void) { return 200000; }
void notify_custom_renderer_unit_state(Unit *unit,unsigned kind) {
    assert(unit&&kind==C3X_RENDERER_UNIT_STATE_OBSERVE);++states;
}
int configured_move(int ox,int oy,int nx,int ny,int value) {
    assert(!((ox+oy)&1)&&!((nx+ny)&1));++configured_calls;last_rings=value;return world_result;
}
int legacy_move(int ox,int oy,int nx,int ny) {
    assert(!((ox+oy)&1)&&!((nx+ny)&1));++legacy_calls;return world_result;
}
int reconcile(void) { ++reconciles;return world_result; }
int movement(struct c3x_renderer_unit_move_v1 const *event) { ++movement_calls;last_move=*event;return 1; }
void reset(void) {
    memset(&state,0,sizeof state);memset(&screen,0,sizeof screen);memset(animator_words,0,sizeof animator_words);
    bic.Map=(Map){100,80,0};screen.TileX_Min=22;screen.TileX_Max=26;screen.TileY_Min=10;screen.TileY_Max=14;
    screen.animator.field_18E4=animator_words;
    state.current_config.enable_custom_rendering=true;state.custom_renderer_viewer_civ_id=1;
    state.custom_renderer_map_epoch=4;state.custom_renderer_viewer_epoch=3;
    state.custom_renderer_qpc_frequency.QuadPart=1000000;state.custom_renderer_visual_clock=clock_ticks;
    state.custom_renderer_world_move_sight=configured_move;state.custom_renderer_world_move=legacy_move;
    state.custom_renderer_world_reconcile=reconcile;state.custom_renderer_unit_move=movement;
    rings=3;world_result=1;target_visible=false;
    configured_calls=legacy_calls=reconciles=movement_calls=states=0;
}
'''

CASES = r'''
void geometry(void) {
    reset();
    // Seven rings reach fourteen raw coordinates; six raw coordinates cannot.
    assert(!custom_renderer_sight_near_view(8,12,3));
    assert(custom_renderer_sight_near_view(8,12,7));
    assert(!custom_renderer_sight_near_view(8,28,7)); // Rectangular corner lies outside actual native closure.
    screen.TileX_Min=0;screen.TileX_Max=4;screen.TileY_Min=0;screen.TileY_Max=4;
    bic.Map.Flags=1;assert(custom_renderer_sight_near_view(98,2,1));
    bic.Map.Flags=2;assert(custom_renderer_sight_near_view(2,78,1));
    bic.Map.Flags=3;assert(custom_renderer_sight_near_view(98,78,2));
    assert(!custom_renderer_sight_near_view(98,78,1));
    for(int r=0;r<=7;++r){
        bic.Map.Flags=0;screen.TileX_Min=40;screen.TileX_Max=42;screen.TileY_Min=40;screen.TileY_Max=40;
        assert(custom_renderer_sight_near_view(40-2*r,40,r));
        if(r)assert(!custom_renderer_sight_near_view(40-2*r,40,r-1));
    }
    assert(!custom_renderer_sight_near_view(0,0,-1)&&!custom_renderer_sight_near_view(0,0,8));
}
void callback(void) {
    reset();Unit unit={.Body={.ID=9,.X=8,.Y=12,.CivID=1}};rings=7;
    notify_custom_renderer_unit_move(&unit,6,12,false);
    assert(configured_calls==1&&last_rings==7&&movement_calls==0&&states==1);
    assert(state.custom_renderer_redraw_pending&&(state.custom_renderer_dirty_flags&C3X_RENDERER_DIRTY_SCENE));
    reset();unit.Body.CivID=2;rings=7;
    notify_custom_renderer_unit_move(&unit,6,12,false);
    assert(configured_calls==0&&movement_calls==0&&states==1&&!state.custom_renderer_redraw_pending);
    // Foreign hidden-to-visible entry gets an ordered accepted event and both sight closures.
    target_visible=true;notify_custom_renderer_unit_move(&unit,6,12,false);
    assert(configured_calls==1&&movement_calls==1&&!last_move.source_visible&&last_move.target_visible);
    assert(last_move.old_x==6&&last_move.new_x==8&&last_move.presentation_time_ticks==200000);
    target_visible=false;notify_custom_renderer_unit_move(&unit,6,12,true);
    assert(configured_calls==2&&movement_calls==2&&last_move.source_visible&&!last_move.target_visible);
    reset();unit.Body.CivID=1;unit.Body.X=70;unit.Body.Y=50;
    notify_custom_renderer_unit_move(&unit,68,50,false);
    assert(configured_calls==1&&!state.custom_renderer_redraw_pending); // Offscreen own sight refresh, no visual work.
    state.current_config.enable_custom_rendering=false;
    notify_custom_renderer_unit_move(&unit,68,50,true);
    assert(configured_calls==1&&movement_calls==0&&states==1);
    state.current_config.enable_custom_rendering=true;
    notify_custom_renderer_unit_move(&unit,70,50,true);
    assert(configured_calls==1&&states==1); // No accepted move.
}
void compatibility(void) {
    reset();state.custom_renderer_world_move_sight=NULL;
    assert(refresh_custom_renderer_sight(6,6,8,6,3)==1&&legacy_calls==1&&!reconciles);
    assert(refresh_custom_renderer_sight(6,6,8,6,7)==1&&legacy_calls==1&&reconciles==1);
    state.custom_renderer_world_reconcile=NULL;
    assert(refresh_custom_renderer_sight(6,6,8,6,7)==C3X_RENDERER_RESULT_ERROR);
    state.custom_renderer_world_move_sight=configured_move;world_result=C3X_RENDERER_RESULT_ERROR;
    Unit unit={.Body={.ID=9,.X=8,.Y=12,.CivID=1}};
    notify_custom_renderer_unit_move(&unit,6,12,false);
    assert(state.custom_renderer_world_audit_needed);
}
int main(int argc,char **argv) {
    assert(argc==2);
    if(!strcmp(argv[1],"geometry"))geometry();
    else if(!strcmp(argv[1],"callback"))callback();
    else if(!strcmp(argv[1],"compatibility"))compatibility();
    else assert(false);
}
'''


class InjectedSightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("Host clang is unavailable")
        cls.temporary = tempfile.TemporaryDirectory(prefix="c3x-sight-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        source = (ROOT / "injected_code.c").read_text()
        native = (ROOT / "ref/Civ3Conquests_master.exe.c").read_text()
        start = native.index("void __cdecl neighbor_index_to_diff(")
        end = native.index("\n}\n", start) + 3
        contract = directory / "contract.c"
        contract.write_text(PRELUDE + native[start:end] + "\n" +
                            "\n".join(production_function(source, name) for name in FUNCTIONS) + CASES)
        cls.executable = directory / "contract"
        result = subprocess.run([compiler, "-std=c17", "-O1", "-Wall", "-Wextra", "-Werror",
                                 "-Wno-unused-parameter", "-Wno-parentheses", "-I", str(ROOT),
                                 str(contract), "-o", str(cls.executable)],
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def contract(self, name):
        result = subprocess.run([str(self.executable), name], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_configured_native_ring_geometry_and_wrapped_view(self):
        self.contract("geometry")

    def test_accepted_movement_visibility_and_config_off(self):
        self.contract("callback")

    def test_older_dll_range_compatibility_is_explicit(self):
        self.contract("compatibility")
