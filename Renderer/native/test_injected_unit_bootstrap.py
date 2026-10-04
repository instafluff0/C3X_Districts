"""Host C contracts for the first native unit-view capture.

Production injected bodies are extracted on each run. The native mocks preserve
the view producer's selection and draw semantics without loading Windows or
dispatching a renderer build.
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = (
    "custom_renderer_native_probe_on",
    "custom_renderer_tile_visible_at",
    "forward_custom_unit_body",
    "patch_Unit_tick_anim",
    "patch_Sprite_draw_unit_body_normal",
    "patch_Sprite_draw_unit_body_reduced",
    "patch_Unit_draw_map_status",
    "patch_Animator_draw_map_unit_cursor",
    "bootstrap_custom_renderer_initial_units",
    "notify_custom_renderer_unit_selection",
)


def production_function(source, name):
    signature = re.search(
        r"^(?:void|bool|int)(?: __fastcall)?\n" + re.escape(name) + r"\s*\(",
        source, re.MULTILINE,
    )
    if signature is None:
        raise AssertionError(f"Production function missing: {name}")
    end = re.search(r"^}\s*$", source[signature.end():], re.MULTILINE)
    if end is None:
        raise AssertionError(f"Production function incomplete: {name}")
    return source[signature.start():signature.end() + end.end()] + "\n"


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Renderer/native/c3x_renderer_api.h"
#define __fastcall
#define __ 0
enum { IS_UNINITED, IS_OK, IS_INIT_FAILED };
enum { AT_BLANK = 0, AT_DEFAULT = 1, AT_RUN = 2, AT_ATTACK1 = 3, AT_DEATH = 6, AT_FIDGET = 8, AT_FORTRESS = 11, AT_ROAD = 13, AT_PLANT = 18 };
enum { DNCM_OFF, SCM_OFF = 0, CS_SUMMER = 0, CS_SPRING = 3, UTA_Army = 1 };
typedef struct RECT { int left, top, right, bottom; } RECT;
typedef struct LARGE_INTEGER { long long QuadPart; } LARGE_INTEGER;
typedef struct Sprite { int Width, Height; } Sprite;
typedef struct FLC_Frame_Image { void *Flic_Info; Sprite sprite; int image_revision; } FLC_Frame_Image;
typedef struct AnimationSummary {
    int current_anim_type, queued_anim_type, direction_2;
    int pixel_loc_x, pixel_loc_y, pixel_target_x, pixel_target_y;
} AnimationSummary;
typedef struct Animation_Info { int *Frame_Counts; float *anim_frame_time_seconds; } Animation_Info;
typedef struct Unit {
    struct {
        int ID, UnitTypeID, X, Y, CivID, Damage, Container_Unit, UnitState;
        int field_23D, field_233, Active, army_top_defender_id;
        bool always_on_top;
        void *field_234;
        RECT Rect;
        struct {
            FLC_Frame_Image Frame_1;
            Animation_Info *Animation_Info;
            AnimationSummary summary;
            int field_FC, field_12C;
            bool field_111;
        } Animation;
    } Body;
    bool army, visible, worker;
    char online_hidden;
} Unit;
typedef struct JGL_Image { int identity; } JGL_Image;
typedef struct PCX_Image { struct { JGL_Image *Image; } JGL; } PCX_Image;
typedef struct JGL_Color_Table JGL_Color_Table;
typedef struct JGL_Vtable {
    int (*m04_Get_Palette_Colors)(JGL_Color_Table *, int, unsigned char *, int, int);
} JGL_Vtable;
struct JGL_Color_Table { JGL_Vtable *vtable; };
typedef struct PCX_Color_Table { JGL_Color_Table *JGL_Color_Table; } PCX_Color_Table;
typedef struct Tile Tile;
typedef struct Tile_Vtable { int (*m45_Get_City_ID)(Tile *); } Tile_Vtable;
struct Tile {
    Tile_Vtable *vtable;
    struct { unsigned FOWStatus, V3, Visibility, field_D0_Visibility; } Body;
    bool visible, leader_visible;
    int city_id;
    Unit *primary, *secondary;
};
typedef struct Map Map;
typedef struct Map_Vtable { bool (*m10_Get_Map_Zoom)(Map *); } Map_Vtable;
struct Map {
    Map_Vtable *vtable;
    int Width, Height, Flags;
    struct { int field_3EA4, field_3EA8, field_3E98[3]; } Renderer;
};
typedef struct Animator { intptr_t field_18E4[22]; } Animator;
typedef struct Main_Screen_Form {
    int Player_CivID, TileX_Min, TileX_Max, TileY_Min, TileY_Max, camera_x, camera_y;
    int Mode_Action;
    Unit *Current_Unit;
    Animator animator;
    struct { struct { PCX_Image Canvas; } Data; } Units_Control;
    struct { PCX_Image Canvas; } Base_Data;
} Main_Screen_Form;
typedef struct Leader { int ID; unsigned Contacts[32]; } Leader;
typedef struct UnitType { char Civilipedia_Entry[32]; } UnitType;
typedef struct Bic { Map Map; int UnitTypeCount; UnitType *UnitTypes; bool is_zoomed_out; } Bic;
typedef struct State {
    struct {
        bool enable_custom_rendering, enable_custom_rendering_zoom, enable_unit_counters;
        int day_night_cycle_mode, seasonal_cycle_mode;
    } current_config;
    int custom_renderer_init_state, custom_renderer_viewer_civ_id;
    long long custom_renderer_display_viewer_epoch, custom_renderer_viewer_epoch;
    bool custom_renderer_native_probe_active;
    void *custom_renderer_native_observe;
    unsigned (*custom_renderer_probe_thread_id)(void);
    unsigned custom_renderer_probe_owner;
    bool custom_renderer_draw_in_progress, custom_renderer_frame_active;
    bool custom_renderer_capture_only, custom_renderer_capture_failed;
    bool custom_renderer_unit_bootstrap, custom_renderer_unit_bootstrap_failed;
    bool custom_renderer_unit_representatives_dirty, custom_renderer_redraw_pending;
    int custom_renderer_unit_display_action;
    unsigned custom_renderer_dirty_flags;
    unsigned custom_renderer_unit_bootstrap_copies;
    Unit *custom_renderer_unit_bootstrap_selected, *custom_renderer_unit_context;
    PCX_Image *custom_renderer_unit_canvas;
    struct c3x_renderer_tile_v1 *custom_renderer_tiles;
    int custom_renderer_tile_count;
    c3x_renderer_unit_draw_background_fn custom_renderer_unit_draw;
    c3x_renderer_unit_visual_fn custom_renderer_unit_visual;
    c3x_renderer_unit_animation_fn custom_renderer_unit_animation;
    c3x_renderer_unit_forget_fn custom_renderer_unit_forget;
    c3x_renderer_native_image_fn custom_renderer_native_image;
    long long (*custom_renderer_visual_clock)(void);
    LARGE_INTEGER custom_renderer_qpc_frequency, custom_renderer_animation_timestamp;
    LARGE_INTEGER custom_renderer_animation_sample_at;
    int custom_renderer_zoom_tile_width, custom_renderer_zoom_native_tile_width;
    bool day_night_cycle_unstarted, seasonal_cycle_unstarted;
    int current_day_night_cycle, current_seasonal_cycle, custom_renderer_test_step;
    char custom_renderer_test_save[1];
} State;
static State state, *is = &state;
static Bic bic, *p_bic_data = &bic;
static Main_Screen_Form screen, *p_main_screen_form = &screen;
static Leader leaders[32];
static Tile tiles[512];
static Unit units[12];
static Unit *top_units[1025];
static struct c3x_renderer_tile_v1 occurrences[12];
static UnitType unit_type;
static int counts[19]; static float frame_seconds[19];
static Animation_Info animation_info;
static JGL_Image image, background_image;
static PCX_Image background;
static PCX_Color_Table palette;
static unsigned preferences, *p_preferences = &preferences;
static unsigned debug_bits, *p_debug_mode_bits = &debug_bits;
static unsigned owner_thread;
static bool online, submission_success, palette_ready, producer_changes_member;
static int tick_count, status_count, cursor_count, hud_scope_count, gui_count, marker_count;
static int original_bodies, observations, forget_count, errors, debug_count, query_count;
static int native_selection_calls; static bool native_selection_accepts;
static int draw_count, animation_count, visual_count, last_edx, last_status_x, last_status_y;
static bool observed_ids[12];
static long long observed_ticks[12];
static int tick_ids[2048], tick_offsets[2048][2], query_xy[2048][2];
static unsigned draw_flags[2048];
static struct c3x_renderer_unit_v1 draws[2048];
static struct c3x_renderer_unit_animation_v1 animations[2048];
static long long qpc;
unsigned current_thread_id(void) { return owner_thread; }
bool is_online_game(void) { return online; }
bool Map_in_range(Map *map, int unused, int x, int y) {
    (void)unused; return x >= 0 && y >= 0 && x < map->Width && y < map->Height;
}
void wrap_tile_coords(Map *map, int *x, int *y) {
    if (map->Flags & 1) *x = (*x % map->Width + map->Width) % map->Width;
    if (map->Flags & 2) *y = (*y % map->Height + map->Height) % map->Height;
}
Tile *tile_at(int x, int y) {
    assert(Map_in_range(&bic.Map, 0, x, y)); return &tiles[y * bic.Map.Width + x];
}
unsigned capture_custom_renderer_visibility(Tile *tile, int viewer, int x, int y) {
    assert(tile == tile_at(x, y) && viewer == 1);
    return C3X_RENDERER_TILE_VISIBILITY_KNOWN | C3X_RENDERER_TILE_EXPLORED |
        (tile->visible ? C3X_RENDERER_TILE_VISIBLE : 0);
}
Unit *patch_Main_Screen_Form_find_visible_unit(Main_Screen_Form *form, int unused,
                                             int x, int y, Unit *excluded) {
    (void)unused; assert(form == &screen);
    Tile *tile = tile_at(x, y);
    if (excluded == NULL) {
        assert(query_count < 2048); query_xy[query_count][0] = x; query_xy[query_count++][1] = y;
        if (producer_changes_member && form->Current_Unit != NULL)
            form->Current_Unit->Body.army_top_defender_id = units[1].Body.ID;
        return tile->primary;
    }
    assert(excluded == tile->primary); return tile->secondary;
}
bool patch_Leader_is_tile_visible(Leader *leader, int unused, int x, int y) {
    (void)unused; assert(leader == &leaders[1]); return tile_at(x, y)->leader_visible;
}
bool patch_Unit_is_visible_to_civ(Unit *unit, int unused, int viewer, int mode) {
    (void)unused; assert(viewer == 1 && mode == 1); return unit->visible;
}
bool Unit_has_ability(Unit *unit, int unused, int ability) {
    (void)unused; assert(ability == UTA_Army); return unit->army;
}
Unit *get_unit_ptr(int id) {
    for (int n = 0; n < 12; ++n) if (units[n].Body.ID == id) return &units[n];
    return NULL;
}
bool is_worker(Unit *unit) { return unit->worker; }
int Unit_get_max_hp(Unit *unit) { (void)unit; return 4; }
int clamp(int minimum, int maximum, int value) {
    return value < minimum ? minimum : value > maximum ? maximum : value;
}
bool QueryPerformanceCounter(LARGE_INTEGER *value) { value->QuadPart = qpc; qpc += 1000; return true; }
long long advancing_visual_clock(void) { qpc += 1000; return qpc; }
bool custom_renderer_zoom_enabled(void) { return state.current_config.enable_custom_rendering_zoom; }
bool custom_renderer_zoom_transform_active(void) { return custom_renderer_zoom_enabled(); }
void sync_custom_renderer_zoom_to_native(void) {}
void custom_renderer_zoom_transform_point(int *x, int *y) {
    if (custom_renderer_zoom_enabled()) { *x = *x * 3 / 4 + 17; *y = *y * 3 / 4 - 11; }
}
int custom_renderer_hud_scope(JGL_Image *value, int x, int y, unsigned owner) {
    (void)value; (void)x; (void)y; (void)owner; ++hud_scope_count; return 1;
}
void Unit_draw_status(Unit *unit, int edx, PCX_Image *canvas, int x, int y, bool stack_marks) {
    (void)unit; (void)canvas; (void)stack_marks;
    ++status_count; last_edx = edx; last_status_x = x; last_status_y = y;
}
void Animator_draw_unit_cursor(Animator *animator, int unused, int x, int y) {
    (void)unused; assert(animator == &screen.animator); ++cursor_count;
    last_status_x = x; last_status_y = y;
}
void notify_custom_renderer_unit_state(Unit *unit, unsigned kind) {
    assert(unit != NULL && kind == C3X_RENDERER_UNIT_STATE_OBSERVE); ++observations;
    assert(unit->Body.ID >= 100 && unit->Body.ID < 112); observed_ids[unit->Body.ID - 100] = true;
    if (state.custom_renderer_visual_clock) observed_ticks[unit->Body.ID - 100] = state.custom_renderer_visual_clock();
}
void Main_Screen_Form_set_selected_unit(Main_Screen_Form *form,int unused,Unit *unit,bool flag) {
    (void)unused;(void)flag;++native_selection_calls;
    if (native_selection_accepts) form->Current_Unit=unit;
}
void forget_unit(int id) { assert(id >= 0); ++forget_count; }
void log_custom_renderer_event(char const *event, int result) {
    assert(event != NULL && result != C3X_RENDERER_RESULT_OK); ++errors;
}
void debug_output(char const *message) { assert(strstr(message, "unit-bootstrap") != NULL); ++debug_count; }
static void (*p_OutputDebugStringA)(char const *) = debug_output;
int colors(JGL_Color_Table *table, int unused, unsigned char *out, int first, int count) {
    (void)table; (void)unused; assert(first == 6 && count == 1);
    out[0] = 12; out[1] = 34; out[2] = 56; return 0;
}
int capture_visual(struct c3x_renderer_unit_visual_v1 const *visual) {
    assert(visual->struct_size == sizeof *visual); ++visual_count; return C3X_RENDERER_RESULT_OK;
}
int capture_animation(struct c3x_renderer_unit_animation_v1 const *animation) {
    assert(animation->struct_size == sizeof *animation && animation_count < 2048);
    assert(animation->visual.presentation_time_ticks >= observed_ticks[animation->visual.unit_id - 100]);
    animations[animation_count++] = *animation; return capture_visual(&animation->visual);
}
int unused_draw(struct c3x_renderer_unit_v1 const *draw, void *destination, void *underlay) {
    (void)draw; (void)destination; (void)underlay; assert(!"Resident native publication expected"); return 0;
}
int translate_custom_renderer_native(int operation, JGL_Image *destination, void *underlay,
                                    RECT *source, RECT *bounds, unsigned flags) {
    assert(operation == C3X_NATIVE_UNIT_DRAW && destination == &image && underlay == &background_image);
    assert(source != NULL && bounds != NULL && draw_count < 2048);
    draws[draw_count] = *(struct c3x_renderer_unit_v1 *)source;
    assert(observed_ids[draws[draw_count].unit_id - 100]); // Every copied body has its own ordered visible state.
    draw_flags[draw_count++] = flags;
    return submission_success ? 1 : -1;
}
int Sprite_draw_unit_body_normal(Sprite *sprite, int unused, PCX_Image *underlay,
                                PCX_Image *canvas, int x, int y, char *path, PCX_Color_Table *table) {
    (void)sprite; (void)unused; (void)underlay; (void)canvas; (void)x; (void)y; (void)path; (void)table;
    ++original_bodies; return 77;
}
int Sprite_draw_unit_body_reduced(Sprite *sprite, int unused, PCX_Image *underlay,
                                 PCX_Image *canvas, int x, int y, int sx, int sy, int divisor,
                                 char *path, PCX_Color_Table *table) {
    assert(sx == 1 && sy == 1 && divisor == 2);
    return Sprite_draw_unit_body_normal(sprite, unused, underlay, canvas, x, y, path, table);
}
void Unit_tick_anim(Unit *, int, PCX_Image *, int, int, bool);
'''


CASES = r'''
bool map_zoom(Map *map) { assert(map == &bic.Map); return bic.is_zoomed_out; }
int city_id(Tile *tile) { return tile->city_id; }
static Map_Vtable map_vtable = { map_zoom };
static Tile_Vtable tile_vtable = { city_id };
static JGL_Vtable palette_vtable = { colors };
static JGL_Color_Table color_table = { &palette_vtable };
void Unit_tick_anim(Unit *unit, int unused, PCX_Image *canvas, int offset_x, int offset_y, bool status) {
    (void)unused; assert(tick_count < 2048);
    tick_ids[tick_count] = unit->Body.ID;
    tick_offsets[tick_count][0] = offset_x; tick_offsets[tick_count++][1] = offset_y;
    if (state.custom_renderer_unit_bootstrap) {
        assert(!status && screen.Current_Unit == NULL && (*p_preferences & 0x800) == 0);
        assert(state.custom_renderer_unit_context == unit && state.custom_renderer_unit_canvas == canvas);
    }
    if (!unit->visible && !(debug_bits & 8)) return;
    if (tile_at(unit->Body.X, unit->Body.Y)->city_id >= 0 && !(*p_preferences & 0x2000) &&
        screen.Current_Unit != unit && !unit->Body.always_on_top &&
        (!unit->Body.Animation.field_111 || unit->Body.Animation.summary.current_anim_type < AT_ATTACK1 ||
         unit->Body.Animation.summary.current_anim_type > AT_DEATH)) return;
    if (*p_preferences & 0x800) ++marker_count;
    if (screen.Current_Unit == unit) {
        ++gui_count;
        patch_Animator_draw_map_unit_cursor(&screen.animator, 0, 123, 456);
    }
    Unit *member = unit->army ? get_unit_ptr(unit->Body.army_top_defender_id) : NULL;
    for (int child = 0; child < (member != NULL ? 2 : 1); ++child) {
        Unit *body = child ? member : unit;
        Sprite *sprite = &body->Body.Animation.Frame_1.sprite;
        int divisor = bic.is_zoomed_out ? 2 : 1;
        int center_x = body->Body.Animation.summary.pixel_loc_x / divisor - offset_x;
        int center_y = body->Body.Animation.summary.pixel_loc_y / divisor - offset_y;
        // Native army commander is displaced forty pixels (twenty at reduced zoom).
        if (member != NULL && child == 0) center_x += 40 / divisor;
        int x = center_x - sprite->Width / (2 * divisor);
        int y = center_y - sprite->Height / (2 * divisor);
        PCX_Color_Table *active_palette = palette_ready ? &palette : NULL;
        AnimationSummary before_draw = body->Body.Animation.summary;
        int before_cursor = body->Body.Animation.field_FC;
        if (bic.is_zoomed_out)
            patch_Sprite_draw_unit_body_reduced(sprite, 0, &background, canvas, x, y, 1, 1, 2, "palette", active_palette);
        else patch_Sprite_draw_unit_body_normal(sprite, 0, &background, canvas, x, y, "palette", active_palette);
        assert(memcmp(&body->Body.Animation.summary, &before_draw, sizeof before_draw) == 0);
        assert(body->Body.Animation.field_FC == before_cursor);
        // Frame acquisition/native army drawing may alter temporary image state.
        // Deliberately perturb each saved field to exercise complete restoration.
        body->Body.Rect = (RECT){-900, -800, 900, 800};
        body->Body.Animation.Frame_1.image_revision += 10;
        body->Body.Animation.summary.pixel_loc_x += 77;
        body->Body.Animation.summary.current_anim_type = AT_DEATH;
        body->Body.Animation.field_FC += 2;
        body->Body.Animation.field_12C += 4;
    }
    patch_Unit_draw_map_status(unit, 71, canvas, 123, 456, status);
}
void reset(void) {
    memset(&state, 0, sizeof state); memset(&bic, 0, sizeof bic); memset(&screen, 0, sizeof screen);
    memset(leaders, 0, sizeof leaders); memset(tiles, 0, sizeof tiles); memset(units, 0, sizeof units);
    memset(occurrences, 0, sizeof occurrences); memset(top_units, 0, sizeof top_units);
    p_main_screen_form = &screen;
    state.current_config.enable_custom_rendering = true;
    state.custom_renderer_init_state = IS_OK; state.custom_renderer_viewer_civ_id = 1;
    state.custom_renderer_viewer_epoch = 9;
    state.custom_renderer_draw_in_progress = state.custom_renderer_frame_active = true;
    state.custom_renderer_native_probe_active = true; state.custom_renderer_native_observe = &image;
    state.custom_renderer_probe_thread_id = current_thread_id;
    state.custom_renderer_probe_owner = owner_thread = 7;
    state.custom_renderer_unit_draw = unused_draw;
    state.custom_renderer_unit_visual = capture_visual;
    state.custom_renderer_unit_animation = capture_animation;
    state.custom_renderer_unit_forget = forget_unit;
    state.custom_renderer_tiles = occurrences; state.custom_renderer_tile_count = 1;
    state.custom_renderer_zoom_native_tile_width = state.custom_renderer_zoom_tile_width = 128;
    state.custom_renderer_qpc_frequency.QuadPart = 1000000;
    bic.Map.vtable = &map_vtable; bic.Map.Width = bic.Map.Height = 8;
    bic.UnitTypeCount = 1; bic.UnitTypes = &unit_type; strcpy(unit_type.Civilipedia_Entry, "PRTO_Archer");
    screen.Player_CivID = 1; screen.TileX_Min = 0; screen.TileX_Max = 8;
    screen.TileY_Min = 0; screen.TileY_Max = 7;
    screen.Units_Control.Data.Canvas.JGL.Image = &image;
    screen.Base_Data.Canvas.JGL.Image = &background_image;
    background.JGL.Image = &background_image;
    palette.JGL_Color_Table = &color_table;
    leaders[1].ID = 1;
    for (int n = 0; n < 19; ++n) { counts[n] = 16; frame_seconds[n] = .0625f; }
    animation_info.Frame_Counts = counts; animation_info.anim_frame_time_seconds = frame_seconds;
    for (int n = 0; n < 512; ++n) {
        tiles[n].vtable = &tile_vtable; tiles[n].visible = tiles[n].leader_visible = true;
        tiles[n].Body.Visibility = 1u << 1; tiles[n].city_id = -1;
    }
    for (int n = 0; n < 12; ++n) {
        Unit *unit = &units[n]; unit->Body.ID = 100 + n; unit->Body.CivID = 2;
        unit->Body.Container_Unit = unit->Body.army_top_defender_id = -1;
        unit->Body.field_234 = &unit->online_hidden; unit->visible = true;
        unit->Body.Rect = (RECT){20 + n, 30 + n, 45 + n, 55 + n};
        unit->Body.Animation.Frame_1.Flic_Info = &animation_info;
        unit->Body.Animation.Frame_1.sprite = (Sprite){191, 183};
        unit->Body.Animation.Animation_Info = &animation_info;
        unit->Body.Animation.summary = (AnimationSummary){AT_RUN, AT_DEFAULT, 1, 0, 0, 10, 20};
        unit->Body.Animation.field_FC = 7 + n; unit->Body.Animation.field_12C = 13 + n;
    }
    preferences = 0x800u; debug_bits = 0; qpc = 1000000;
    online = producer_changes_member = false; submission_success = palette_ready = true;
    tick_count = status_count = cursor_count = hud_scope_count = gui_count = marker_count = 0;
    original_bodies = observations = forget_count = errors = debug_count = query_count = 0;
    draw_count = animation_count = visual_count = native_selection_calls = 0;
    native_selection_accepts = true;
    memset(observed_ids, 0, sizeof observed_ids);
    memset(observed_ticks, 0, sizeof observed_ticks);
}
void place(int index, int x, int y) {
    Unit *unit = &units[index]; unit->Body.X = x; unit->Body.Y = y;
    unit->Body.Animation.summary.pixel_loc_x = x * 64 + 9;
    unit->Body.Animation.summary.pixel_loc_y = y * 32 + 5;
    tile_at(x, y)->primary = unit;
}
void occurrence(int index, int x, int y, int pair) {
    int width = bic.is_zoomed_out ? 64 : 128;
    occurrences[index] = (struct c3x_renderer_tile_v1){0};
    occurrences[index].tile_flags = C3X_RENDERER_TILE_RENDER;
    occurrences[index].tile_x = x; occurrences[index].tile_y = y;
    occurrences[index].anchor_x = x * width / 2 - (int)screen.animator.field_18E4[pair];
    occurrences[index].anchor_y = y * width / 4 - (int)screen.animator.field_18E4[pair + 1];
    custom_renderer_zoom_transform_point(&occurrences[index].anchor_x, &occurrences[index].anchor_y);
    if (state.custom_renderer_tile_count <= index) state.custom_renderer_tile_count = index + 1;
}
void restored(Unit const *before, Unit const *after) {
    assert(memcmp(&before->Body.Rect, &after->Body.Rect, sizeof(RECT)) == 0);
    assert(memcmp(&before->Body.Animation.summary, &after->Body.Animation.summary, sizeof(AnimationSummary)) == 0);
    assert(memcmp(&before->Body.Animation.Frame_1, &after->Body.Animation.Frame_1, sizeof(FLC_Frame_Image)) == 0);
    assert(before->Body.Animation.field_FC == after->Body.Animation.field_FC);
    assert(before->Body.Animation.field_12C == after->Body.Animation.field_12C);
}
void clean_scope(Unit *selected, unsigned saved_preferences) {
    assert(screen.Current_Unit == selected && preferences == saved_preferences);
    assert(!state.custom_renderer_unit_bootstrap && state.custom_renderer_unit_bootstrap_selected == NULL);
    assert(status_count == 0 && cursor_count == 0 && hud_scope_count == 0 && gui_count == 0 && marker_count == 0);
}
void order_case(void) {
    reset(); place(0, 6, 0); place(1, 4, 0); place(2, 2, 0); place(3, 1, 1);
    units[0].Body.Active = 0x100;
    units[4].Body.X = 6; units[4].Body.Y = 0;
    units[4].Body.Animation.summary.pixel_loc_x = 6 * 64 + 9;
    tile_at(6, 0)->secondary = &units[4];
    units[2].Body.always_on_top = true;
    top_units[0] = NULL; top_units[1] = &units[2];
    screen.animator.field_18E4[7] = (intptr_t)top_units;
    screen.animator.field_18E4[8] = (intptr_t)(top_units + 2);
    occurrence(0, 6, 0, 14); occurrence(1, 4, 0, 14);
    occurrence(2, 2, 0, 14); occurrence(3, 1, 1, 14);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    int expected[] = {104, 100, 101, 103, 102};
    assert(tick_count == 5 && draw_count == 5 && state.custom_renderer_unit_bootstrap_copies == 5);
    assert(memcmp(tick_ids, expected, sizeof expected) == 0);
    assert(query_count == 32);
    int query = 0;
    for (int y = 0; y < 8; ++y) for (int x = 7; x >= 0; --x) if (!((x + y) & 1)) {
        assert(query_xy[query][0] == x && query_xy[query++][1] == y);
    }
    assert(original_bodies == 0 && observations == 5 && animation_count == 5);
    clean_scope(NULL, 0x800u);
}
void eligibility_case(void) {
    for (int condition = 0; condition < 13; ++condition) {
        reset(); place(0, 2, 2); occurrence(0, 2, 2, 14);
        bool expected = false;
        switch (condition) {
        case 0: units[0].Body.field_23D = 1; break;
        case 1: online = true; units[0].online_hidden = 1; break;
        case 2: tile_at(2, 2)->visible = false; break;
        case 3: tile_at(2, 2)->Body.Visibility = 0; break;
        case 4: units[0].visible = false; break;
        case 5: tile_at(2, 2)->city_id = 42; break;
        case 6: tile_at(2, 2)->city_id = 42; units[0].Body.Animation.field_111 = true;
                units[0].Body.Animation.summary.current_anim_type = AT_ATTACK1; expected = true; break;
        case 7: tile_at(2, 2)->city_id = 42; screen.Current_Unit = &units[0]; expected = true; break;
        case 8: tile_at(2, 2)->Body.Visibility = 0; leaders[1].Contacts[2] = 0x40; expected = true; break;
        case 9: tile_at(2, 2)->Body.Visibility = 0; units[0].Body.field_233 = 1; expected = true; break;
        case 10: tile_at(2, 2)->Body.Visibility = 0; debug_bits = 8; expected = true; break;
        case 11: tile_at(2, 2)->city_id = 42; preferences |= 0x2000; expected = true; break;
        case 12: online = true; units[0].online_hidden = 1; debug_bits = 8; break;
        }
        Unit before = units[0]; Unit *selected = screen.Current_Unit; unsigned saved_preferences = preferences;
        assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
        assert(tick_count == (expected ? 1 : 0) && draw_count == tick_count);
        restored(&before, &units[0]); clean_scope(selected, saved_preferences);
    }
    for (int condition = 0; condition < 5; ++condition) {
        reset(); place(0, 2, 2); occurrence(0, 2, 2, 14); units[0].Body.Active = 0x100;
        units[1].Body.X = units[1].Body.Y = 2; tile_at(2, 2)->secondary = &units[1];
        switch (condition) {
        case 0: units[1].Body.Container_Unit = 100; break;
        case 1: tile_at(2, 2)->leader_visible = false; break;
        case 2: tile_at(2, 2)->leader_visible = false; leaders[1].Contacts[2] = 0x40; break;
        case 3: tile_at(2, 2)->leader_visible = false; debug_bits = 8; break;
        case 4: break;
        }
        assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
        assert(tick_count == (condition < 2 ? 1 : 2));
        assert(tick_ids[tick_count - 1] == 100);
        if (condition >= 2) assert(tick_ids[0] == 101);
    }
}
void anchors_case(void) {
    for (int flags = 0; flags < 4; ++flags) for (int zoom = 0; zoom < 2; ++zoom) {
        reset(); bic.Map.Flags = flags; bic.is_zoomed_out = zoom != 0;
        screen.camera_x = screen.camera_y = 10000;
        for (int pair = 14; pair <= 20; pair += 2) {
            screen.animator.field_18E4[pair] = 101 + pair;
            screen.animator.field_18E4[pair + 1] = 201 + pair;
        }
        int xy[4][2] = {{6, 6}, {2, 6}, {6, 2}, {2, 2}};
        int expected_pair[4];
        for (int n = 0; n < 4; ++n) {
            expected_pair[n] = flags == 3 ? (int[]){14, 16, 18, 20}[n] :
                flags == 1 ? (n == 1 || n == 3 ? 16 : 14) :
                flags == 2 ? (n >= 2 ? 16 : 14) : 14;
            place(n, xy[n][0], xy[n][1]); occurrence(n, xy[n][0], xy[n][1], expected_pair[n]);
        }
        assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
        assert(tick_count == 4 && draw_count == 4);
        for (int n = 0; n < 4; ++n) {
            int id = tick_ids[n] - 100, pair = expected_pair[id], divisor = zoom ? 2 : 1;
            assert(tick_offsets[n][0] == 101 + pair && tick_offsets[n][1] == 201 + pair);
            assert(draws[n].body_x == (xy[id][0] * 64 + 9) / divisor - (101 + pair) - 191 / (2 * divisor));
            assert(draws[n].body_y == (xy[id][1] * 32 + 5) / divisor - (201 + pair) - 183 / (2 * divisor));
            assert(draws[n].reduced == zoom && draws[n].projection_scale_milli == (zoom ? 500 : 1000));
            assert(animations[n].visual.pixel_x == xy[id][0] * 64 + 9);
        }
    }
    reset(); place(0, 2, 2); occurrence(0, 2, 2, 14); occurrences[0].anchor_x++;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_PENDING);
    assert(tick_count == 0 && draw_count == 0); clean_scope(NULL, 0x800u);
    reset(); place(0, 2, 2); state.current_config.enable_custom_rendering_zoom = true;
    state.custom_renderer_zoom_tile_width = 96; occurrence(0, 2, 2, 14);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    int expected_x = (2 * 64 + 9 - 191 / 2) * 3 / 4 + 17;
    assert(draws[0].body_x == expected_x && draws[0].projection_scale_milli == 750);
    // A wrapped view occurrence is resolved by the native map coordinates.
    reset(); bic.Map.Flags = 3; screen.TileX_Min = -2; screen.TileX_Max = 2;
    screen.TileY_Min = -2; screen.TileY_Max = 0; place(0, 6, 6); occurrence(0, 6, 6, 14);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(tick_count == 1 && tick_ids[0] == 100);
}
void readiness_case(void) {
    for (int member = 0; member < 2; ++member) for (int condition = 0; condition < 8; ++condition) {
        reset(); place(0, 2, 2); occurrence(0, 2, 2, 14); screen.Current_Unit = &units[0];
        units[0].army = member != 0; units[0].Body.army_top_defender_id = units[1].Body.ID;
        Unit *target = &units[member];
        switch (condition) {
        case 0: target->Body.Animation.Frame_1.Flic_Info = NULL; break;
        case 1: target->Body.Animation.Animation_Info = NULL; break;
        case 2: animation_info.Frame_Counts = NULL; break;
        case 3: target->Body.Animation.Frame_1.sprite.Width = 0; break;
        case 4: target->Body.Animation.Frame_1.sprite.Height = 0; break;
        case 5: target->Body.Animation.summary.current_anim_type = AT_DEFAULT - 1; break;
        case 6: target->Body.Animation.summary.current_anim_type = AT_PLANT + 1; break;
        case 7: counts[AT_RUN] = 0; break;
        }
        assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_PENDING);
        assert(tick_count == 0 && draw_count == 0 && !state.custom_renderer_unit_bootstrap_failed);
        clean_scope(&units[0], 0x800u);
    }
}
void army_restore_case(void) {
    reset(); state.custom_renderer_visual_clock = advancing_visual_clock; place(0, 2, 2); occurrence(0, 2, 2, 14); screen.Current_Unit = &units[0];
    tile_at(2, 2)->city_id = 42; units[0].army = true;
    units[0].Body.army_top_defender_id = -1; producer_changes_member = true;
    units[1].Body.X = units[1].Body.Y = 2;
    units[1].Body.Animation.summary.pixel_loc_x = 2 * 64 + 13;
    units[1].Body.Animation.summary.pixel_loc_y = 2 * 32 + 7;
    Unit commander = units[0], member = units[1];
    state.custom_renderer_unit_context = &units[9]; state.custom_renderer_unit_canvas = &background;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(tick_count == 1 && draw_count == 2 && state.custom_renderer_unit_bootstrap_copies == 2);
    assert(draws[0].unit_id == 100 && draws[1].unit_id == 101);
    assert(observations == 2 && observed_ids[0] && observed_ids[1]);
    assert(animations[0].display_unit_id == 100 && animations[1].display_unit_id == 100);
    for (int n = 0; n < 2; ++n) {
        assert((draw_flags[n] & C3X_RENDERER_UNIT_SELECTED) != 0);
        assert(draw_flags[n] & C3X_RENDERER_UNIT_CURSOR);
        assert(draws[n].display_color_rgb == 0x0c2238 && !strcmp(draws[n].unit_key, "PRTO_Archer"));
        assert(animations[n].cursor == 7 + n && animations[n].frame_seconds == .0625f);
    }
    restored(&commander, &units[0]); restored(&member, &units[1]);
    assert(units[0].Body.army_top_defender_id == -1);
    assert(state.custom_renderer_unit_context == &units[9] && state.custom_renderer_unit_canvas == &background);
    clean_scope(&units[0], 0x800u);
    state.custom_renderer_unit_bootstrap = true;
    patch_Unit_draw_map_status(&units[0], 99, &screen.Units_Control.Data.Canvas, 11, 23, true);
    patch_Animator_draw_map_unit_cursor(&screen.animator, 99, 11, 23);
    assert(status_count == 0 && cursor_count == 0 && hud_scope_count == 0);
}
void failure_config_case(void) {
    for (int invalid_palette = 0; invalid_palette < 2; ++invalid_palette) {
        reset(); place(0, 2, 2); occurrence(0, 2, 2, 14); screen.Current_Unit = &units[0];
        Unit before = units[0]; submission_success = invalid_palette != 0; palette_ready = invalid_palette == 0;
        state.custom_renderer_unit_context = &units[9]; state.custom_renderer_unit_canvas = &background;
        assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_ERROR);
        assert(tick_count == 1 && state.custom_renderer_unit_bootstrap_failed && original_bodies == 0);
        assert(state.custom_renderer_unit_bootstrap_copies == 0);
        restored(&before, &units[0]); clean_scope(&units[0], 0x800u);
        assert(state.custom_renderer_unit_context == &units[9] && state.custom_renderer_unit_canvas == &background);
    }
    reset(); place(0, 2, 2); occurrence(0, 2, 2, 14); screen.Current_Unit = &units[0];
    state.current_config.enable_custom_rendering = false;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(tick_count == 0 && query_count == 0 && debug_count == 0);
    patch_Unit_tick_anim(&units[0], 0, &screen.Units_Control.Data.Canvas, 11, 23, true);
    assert(tick_count == 1 && original_bodies == 1 && draw_count == 0 && gui_count == 1 && marker_count == 1);
    assert(status_count == 1 && cursor_count == 1 && hud_scope_count == 0);
    assert(state.custom_renderer_unit_context == NULL && state.custom_renderer_unit_canvas == NULL);
    patch_Unit_draw_map_status(&units[0], 99, &screen.Units_Control.Data.Canvas, 11, 23, true);
    assert(status_count == 2 && last_edx == 99 && last_status_x == 11 && last_status_y == 23);
    patch_Animator_draw_map_unit_cursor(&screen.animator, 0, 31, 47);
    assert(cursor_count == 2 && last_status_x == 31 && last_status_y == 47);
    // Ordinary visible tick keeps the cursor flag; hidden ticks never enter native drawing.
    reset(); place(0, 2, 2); screen.Current_Unit = &units[0];
    patch_Unit_tick_anim(&units[0], 0, &screen.Units_Control.Data.Canvas, 0, 0, true);
    assert(draw_count == 1 && (draw_flags[0] & C3X_RENDERER_UNIT_CURSOR));
    reset(); place(0, 2, 2); tiles[2 * bic.Map.Width + 2].visible = false;
    units[0].army = true; units[0].Body.army_top_defender_id = 101;
    patch_Unit_tick_anim(&units[0], 0, &screen.Units_Control.Data.Canvas, 0, 0, true);
    assert(tick_count == 0 && draw_count == 0 && forget_count == 1);
}
void selection_refresh_case(void) {
    reset(); place(0, 2, 2); occurrence(0, 2, 2, 14);
    state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch;
    state.custom_renderer_unit_display_action = screen.Mode_Action;
    state.current_config.enable_unit_counters = true;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(query_count == 0 && draw_count == 0);
    // The accepted attacker changes without either stack moving. Its authoritative
    // native representative changes at the same anchor before the next capture.
    screen.Current_Unit = &units[8];
    place(1, 2, 2); notify_custom_renderer_unit_selection(true);
    assert(state.custom_renderer_unit_representatives_dirty && state.custom_renderer_redraw_pending);
    assert(state.custom_renderer_dirty_flags & C3X_RENDERER_DIRTY_SCENE);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(draw_count == 1 && draws[0].unit_id == 101 && !state.custom_renderer_unit_representatives_dirty);
    int before = query_count;
    notify_custom_renderer_unit_selection(false);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK && query_count == before);
    // A bombard/precision mode can select another representative without movement.
    screen.Mode_Action = 7; place(2, 2, 2); notify_custom_renderer_unit_selection(false);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(draw_count == 2 && draws[1].unit_id == 102);
    // Native frame availability defers exactly once; the request survives retry.
    screen.Mode_Action = 8; notify_custom_renderer_unit_selection(false);
    units[2].Body.Animation.Frame_1.Flic_Info = NULL;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_PENDING);
    assert(state.custom_renderer_unit_representatives_dirty && draw_count == 2);
    units[2].Body.Animation.Frame_1.Flic_Info = &animation_info;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(!state.custom_renderer_unit_representatives_dirty && draw_count == 3);
    // Hidden stacks remain excluded even during this explicit refresh.
    tile_at(2, 2)->visible = false; notify_custom_renderer_unit_selection(true);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK && draw_count == 3);
    // Camera entry requests the same producer only after new anchors are captured.
    tile_at(2, 2)->visible = true; state.custom_renderer_unit_representatives_dirty = true;
    state.custom_renderer_capture_only = true;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_PENDING && draw_count == 3);
    state.custom_renderer_capture_only = false;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK && draw_count == 4);
    // Configuration off leaves native selection/UI ownership untouched.
    state.current_config.enable_custom_rendering = false;
    state.custom_renderer_unit_representatives_dirty = state.custom_renderer_redraw_pending = false;
    state.custom_renderer_dirty_flags = 0; screen.Mode_Action = 9;
    notify_custom_renderer_unit_selection(true);
    assert(!state.custom_renderer_unit_representatives_dirty && !state.custom_renderer_redraw_pending && !state.custom_renderer_dirty_flags);
}
void accepted_selection_case(void) {
    reset();state.custom_renderer_unit_display_action=screen.Mode_Action;
    screen.Current_Unit=&units[0];native_selection_accepts=false;
    execute_native_selection_hook(&screen, &units[1], true);
    assert(native_selection_calls==1&&screen.Current_Unit==&units[0]);
    assert(!state.custom_renderer_unit_representatives_dirty&&!state.custom_renderer_redraw_pending);
    native_selection_accepts=true;
    execute_native_selection_hook(&screen, &units[1], true);
    assert(native_selection_calls==2&&screen.Current_Unit==&units[1]);
    assert(state.custom_renderer_unit_representatives_dirty&&state.custom_renderer_redraw_pending);
    state.custom_renderer_unit_representatives_dirty=state.custom_renderer_redraw_pending=false;
    execute_native_selection_hook(&screen, &units[1], true);
    assert(native_selection_calls==3&&!state.custom_renderer_unit_representatives_dirty);
    execute_native_selection_hook(&screen, NULL, false);
    assert(native_selection_calls==4&&screen.Current_Unit==NULL&&state.custom_renderer_unit_representatives_dirty);
    state.current_config.enable_custom_rendering=false;
    state.custom_renderer_unit_representatives_dirty=state.custom_renderer_redraw_pending=false;
    execute_native_selection_hook(&screen, &units[0], true);
    assert(native_selection_calls==5&&screen.Current_Unit==&units[0]);
    assert(!state.custom_renderer_unit_representatives_dirty&&!state.custom_renderer_redraw_pending);
}
void idle_worker_case(void) {
    // Ordinary draws and first-frame recapture must agree, without modifying
    // native work orders, action cursors or queued actions.
    for (int bootstrap = 0; bootstrap < 2; ++bootstrap)
    for (int action = AT_DEFAULT; action <= AT_PLANT; ++action)
    for (int condition = 0; condition < 6; ++condition) {
        if (action == AT_FIDGET && condition == 4) continue; // Invalid native frame: readiness contract covers deferral.
        reset(); place(0, 2, 2); occurrence(0, 2, 2, 14);
        Unit *unit = &units[0]; unit->worker = condition != 1;
        screen.Current_Unit = condition == 2 ? NULL : unit;
        unit->Body.UnitState = condition == 3 ? 4 : 0;
        unit->Body.Animation.summary.current_anim_type = action;
        unit->Body.Animation.summary.queued_anim_type = AT_PLANT;
        if (condition == 4) counts[AT_FIDGET] = 0;
        if (condition == 5) frame_seconds[AT_FIDGET] = 0;
        Unit before = *unit;
        bool idle = condition != 1 && condition != 2 && condition != 3 &&
            (action == AT_DEFAULT || action == AT_FORTRESS || action >= AT_ROAD);
        int expected = idle ? (condition >= 4 ? AT_DEFAULT : AT_FIDGET) : action;
        if (bootstrap) assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
        else patch_Unit_tick_anim(unit, 0, &screen.Units_Control.Data.Canvas, 0, 0, true);
        assert(draw_count == 1 && animation_count == 1);
        assert(draws[0].action == expected && animations[0].visual.action == expected);
        assert(draws[0].frame_count == counts[expected] && animations[0].frames == counts[expected]);
        assert(draws[0].action_cursor == (idle ? 0 : before.Body.Animation.field_FC));
        assert(draws[0].queued_action == (idle ? AT_BLANK : AT_PLANT));
        assert(animations[0].frame_seconds == frame_seconds[expected]);
        assert(unit->Body.UnitState == before.Body.UnitState);
        if (bootstrap) { restored(&before, unit); clean_scope(screen.Current_Unit, 0x800u); }
    }
    for (int disabled = 0; disabled < 2; ++disabled) {
        reset(); place(0, 2, 2); occurrence(0, 2, 2, 14); screen.Current_Unit = &units[0];
        screen.animator.field_18E4[12] = disabled;
        assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
        assert(draw_flags[0] & C3X_RENDERER_UNIT_SELECTED);
        assert(!!(draw_flags[0] & C3X_RENDERER_UNIT_CURSOR) == !disabled);
        assert(cursor_count == 0); // Selection metadata is copied; native cursor is not redrawn.
    }
}
void guards_case(void) {
    for (int condition = 0; condition < 15; ++condition) {
        reset(); place(0, 2, 2); occurrence(0, 2, 2, 14);
        int expected = C3X_RENDERER_RESULT_PENDING;
        switch (condition) {
        case 0: p_main_screen_form = NULL; break;
        case 1: state.custom_renderer_native_probe_active = false; break;
        case 2: owner_thread = 8; break;
        case 3: state.custom_renderer_draw_in_progress = false; break;
        case 4: state.custom_renderer_frame_active = false; break;
        case 5: state.custom_renderer_capture_only = true; break;
        case 6: state.custom_renderer_capture_failed = true; break;
        case 7: state.custom_renderer_unit_bootstrap = true; break;
        case 8: state.custom_renderer_tile_count = 0; break;
        case 9: state.custom_renderer_tiles = NULL; break;
        case 10: state.custom_renderer_unit_draw = NULL; break;
        case 11: screen.Units_Control.Data.Canvas.JGL.Image = NULL; break;
        case 12: screen.TileX_Max = 257; expected = C3X_RENDERER_RESULT_BAD_ARGUMENT; break;
        case 13: screen.TileY_Max = -2; expected = C3X_RENDERER_RESULT_BAD_ARGUMENT; break;
        case 14: screen.Player_CivID = 32; expected = C3X_RENDERER_RESULT_BAD_ARGUMENT; break;
        }
        assert(bootstrap_custom_renderer_initial_units() == expected);
        assert(tick_count == 0 && draw_count == 0 && query_count == 0);
    }
    for (int condition = 0; condition < 3; ++condition) {
        reset(); screen.animator.field_18E4[7] = (intptr_t)top_units;
        screen.animator.field_18E4[8] = condition == 0 ? (intptr_t)top_units - 1 :
            condition == 1 ? (intptr_t)top_units + 1 : (intptr_t)(top_units + 1025);
        assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_BAD_ARGUMENT);
        assert(tick_count == 0 && draw_count == 0); clean_scope(NULL, 0x800u);
    }
    reset(); state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch;
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_OK);
    assert(query_count == 0 && tick_count == 0);
    // A real top-list overflows the bounded representative list before native drawing.
    reset(); for (int n = 0; n < 1024; ++n) top_units[n] = &units[0];
    place(0, 2, 2); screen.animator.field_18E4[7] = (intptr_t)top_units;
    screen.animator.field_18E4[8] = (intptr_t)(top_units + 1024);
    assert(bootstrap_custom_renderer_initial_units() == C3X_RENDERER_RESULT_BAD_ARGUMENT);
    assert(tick_count == 0 && draw_count == 0); clean_scope(NULL, 0x800u);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    if (!strcmp(argv[1], "order")) order_case();
    else if (!strcmp(argv[1], "eligibility")) eligibility_case();
    else if (!strcmp(argv[1], "anchors")) anchors_case();
    else if (!strcmp(argv[1], "readiness")) readiness_case();
    else if (!strcmp(argv[1], "army_restore")) army_restore_case();
    else if (!strcmp(argv[1], "failure_config")) failure_config_case();
    else if (!strcmp(argv[1], "guards")) guards_case();
    else if (!strcmp(argv[1], "idle_worker")) idle_worker_case();
    else if (!strcmp(argv[1], "selection_refresh")) selection_refresh_case();
    else if (!strcmp(argv[1], "accepted_selection")) accepted_selection_case();
    else assert(!"Unknown contract case");
    return 0;
}
'''


class InjectedUnitBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("Host clang is unavailable")
        cls.temporary = tempfile.TemporaryDirectory(prefix="c3x-unit-bootstrap-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        source = (ROOT / "injected_code.c").read_text()
        contract = directory / "contract.c"
        native_hook = production_function(source, "patch_Main_Screen_Form_set_selected_unit")
        start = native_hook.index("Unit * previous_selected = this->Current_Unit;")
        end = native_hook.index("\n\n", start)
        selected_capture = "void execute_native_selection_hook(Main_Screen_Form *this, Unit *unit, bool param_2) {\n" + native_hook[start:end] + "\n}\n"
        contract.write_text(PRELUDE + "\n".join(production_function(source, name) for name in FUNCTIONS) + selected_capture + CASES)
        cls.executable = directory / "contract"
        result = subprocess.run(
            [compiler, "-std=c17", "-O1", "-Wall", "-Wextra", "-Werror",
             "-Wno-unused-parameter", "-Wno-pointer-to-int-cast", "-Wno-tautological-pointer-compare", "-I", str(ROOT),
             str(contract), "-o", str(cls.executable)],
            text=True, capture_output=True, timeout=30,
        )
        if result.returncode:
            raise AssertionError("Host C contract compile failed:\n" + result.stdout + result.stderr)

    def contract(self, name):
        result = subprocess.run([str(self.executable), name], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_view_order_primary_secondary_and_top_list(self):
        self.contract("order")

    def test_native_representative_and_city_body_eligibility(self):
        self.contract("eligibility")

    def test_wrap_offsets_native_anchors_and_animated_pixel_positions(self):
        self.contract("anchors")

    def test_missing_native_frames_defer_without_drawing(self):
        self.contract("readiness")

    def test_army_body_copy_restores_native_state_and_suppresses_gui(self):
        self.contract("army_restore")

    def test_capture_failures_restore_scope_and_config_off_delegates(self):
        self.contract("failure_config")

    def test_actual_selection_hook_observes_native_acceptance_and_config_off(self):
        self.contract("accepted_selection")

    def test_selection_action_mode_and_camera_refresh_is_one_shot(self):
        self.contract("selection_refresh")

    def test_selected_idle_worker_fidgets_and_bootstrap_keeps_cursor(self):
        self.contract("idle_worker")

    def test_bootstrap_guards_and_bounded_top_list(self):
        self.contract("guards")

    def test_native_draw_entry_has_no_animation_advance_call(self):
        native = (ROOT / "ref/Civ3Conquests_master.exe.c").read_text()
        start = native.index("void __thiscall Unit::tick_anim(")
        end = native.index("void __thiscall FUN_005cc430(", start)
        body = native[start:end]
        # These are actual decompiled callees, separating sprite/frame acquisition
        # from the Animator action tick that the bootstrap must never replay.
        self.assertIn("FLC_Frame_Image::get_frame_image", body)
        self.assertIn("Sprite::FUN_005f88b0", body)
        self.assertIn("Sprite::FUN_005f8940", body)
        self.assertIn("FUN_005cc430(this", body)
        calls = set(re.findall(r"\b([A-Za-z_]\w*(?:::\w+)*)\s*\(", body[body.index("{"):]))
        allowed = {
            "Animator::FUN_004f03e0", "FLC_Frame_Image::get_frame_image",
            "FUN_00403040", "FUN_005b21b0", "FUN_005ba750", "FUN_005cc430",
            "FUN_005f84b0", "Leader::get_color_", "Main_Screen_Form::FUN_004dd800",
            "Map::get_tile", "Sprite::FUN_005f88b0", "Sprite::FUN_005f8940",
            "Tile::has_city", "can_perform_command", "has_ability", "is_online_game",
            "is_visible_to_civ", "memmove", "if", "while",
        }
        self.assertFalse(calls - allowed, f"Unaudited native draw callee: {calls - allowed}")


if __name__ == "__main__":
    unittest.main()
