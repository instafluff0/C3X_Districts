"""Execute injected local city and initial-world contracts with host C mocks.

This harness uses the host clang directly. It never dispatches a Windows or VM
build, and extracts production function bodies on every run.
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
    "capture_custom_renderer_city_body",
    "custom_renderer_tile_visible_at",
    "custom_renderer_tile_near_view",
    "custom_renderer_sight_near_view",
    "refresh_custom_renderer_sight",
    "notify_custom_renderer_city_change",
    "patch_City_recompute_yields_and_happiness",
    "patch_City_update_culture",
    "capture_custom_renderer_world_page",
    "seed_custom_renderer_initial_world",
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Renderer/native/c3x_renderer_api.h"
#define __fastcall
#define __ 0
enum { IS_UNINITED, IS_OK, IS_INIT_FAILED };
typedef struct City {
    struct {
        int ID, X, Y, CivID;
        struct { int Size; } Population;
        int cultural_level, CultureIncome, Total_Cultures[32], FoodIncome;
    } Body;
} City;
typedef struct Tile {
    struct { unsigned Fog_Of_War, FOWStatus, V3, Visibility, field_D0_Visibility; } Body;
} Tile;
typedef struct Map_Renderer { int native_identity; } Map_Renderer;
typedef struct Map {
    int Width, Height, Flags;
    Tile **Tiles;
    Map_Renderer Renderer;
} Map;
typedef struct Leader { int RaceID, Era, CapitalID; } Leader;
typedef struct Race { int CultureGroupID; char CountryName[32], SingularName[32]; } Race;
typedef struct Era { struct { char S[32]; } Name; } Era;
typedef struct Improvement { int Combat_Bombard; } Improvement;
typedef struct Bic {
    Map Map;
    struct { int MaximumSize_City, MaximumSize_Town; } General;
    int RacesCount, ErasCount, ImprovementsCount;
    Race *Races; Era *Eras; Improvement *Improvements;
} Bic;
typedef struct Screen {
    int Player_CivID, TileX_Min, TileX_Max, TileY_Min, TileY_Max;
    bool is_now_loading_game;
    struct { int *field_18E4; } animator;
} Screen;
typedef struct State {
    struct {
        bool enable_custom_rendering, enable_districts, enable_natural_wonders;
        bool enable_distribution_hub_districts;
    } current_config;
    bool distribution_hub_refresh_in_progress, distribution_hub_totals_dirty;
    int custom_renderer_init_state, custom_renderer_viewer_civ_id;
    long long custom_renderer_viewer_epoch, custom_renderer_display_viewer_epoch;
    long long custom_renderer_seeded_viewer_epoch;
    long long custom_renderer_map_epoch, custom_renderer_visibility_revision;
    long long custom_renderer_world_topology_revision;
    struct c3x_renderer_tile_v1 *custom_renderer_tiles;
    int custom_renderer_tile_count, custom_renderer_world_topology_count;
    unsigned *custom_renderer_world_topology;
    unsigned long long *custom_renderer_world_visibility;
    bool custom_renderer_world_audit_needed, custom_renderer_redraw_pending;
    unsigned custom_renderer_dirty_flags;
    c3x_renderer_world_change_fn custom_renderer_world_change;
    c3x_renderer_world_move_fn custom_renderer_world_move;
    c3x_renderer_world_move_sight_fn custom_renderer_world_move_sight;
    c3x_renderer_world_reconcile_fn custom_renderer_world_reconcile;
    c3x_renderer_seed_world_fn custom_renderer_seed_world;
    bool custom_renderer_initial_world_capture, custom_renderer_capture_world_topology;
    bool custom_renderer_native_probe_active;
    void *custom_renderer_native_observe;
    unsigned (*custom_renderer_probe_thread_id)(void);
    unsigned custom_renderer_probe_owner;
    bool custom_renderer_draw_in_progress, custom_renderer_frame_active;
    bool custom_renderer_capture_only, custom_renderer_capture_failed;
    bool custom_renderer_display_valid;
    Map_Renderer *custom_renderer_target;
} State;
static State state, *is = &state;
static Bic bic, *p_bic_data = &bic;
static Screen screen, *p_main_screen_form = &screen;
static Leader leaders[32];
static Race races[2]; static Era eras[2]; static Improvement improvements[2];
static Tile native_tiles[320], null_tile, *p_null_tile = &null_tile;
static Tile *native_tile_ptrs[320];
static struct c3x_renderer_tile_v1 captured[2], output[128];
static unsigned topology[320], debug_bits, *p_debug_mode_bits = &debug_bits;
static unsigned long long visibility_values[320];
static int animator_words[22];
static City city;
static bool visible, explored, wall_active, online;
static unsigned thread_id;
static int wall_reads, raw_reads, copied_records, changed_locations, reconciliations, sight_moves;
static int native_yields, native_culture, distribution_refreshes, cultural_recomputes;
static int world_result, reconcile_result, culture_bonus, native_level, final_level;
static int seed_calls, seed_result, seed_pages, log_calls, logged_result;
static int last_changed_x, last_changed_y, read_failure_at;
static int hidden_delta_reads, initial_appearance_reads;
static struct c3x_renderer_frame_v1 frame;
static struct c3x_renderer_camera_request_v1 request;
int calc_max_visibility_range(void) { return 3; }
void neighbor_index_to_diff(int n, int *x, int *y) {
    if (!n) { *x = *y = 0; return; }
    int r = 1; while ((2*r+1)*(2*r+1) <= n) ++r;
    int k = n - (2*r-1)*(2*r-1);
    int edge = k / (2*r), step = k % (2*r);
    int u = edge == 0 ? -r + step : edge == 1 ? r : edge == 2 ? r-step : -r;
    int v = edge == 0 ? -r : edge == 1 ? -r + step : edge == 2 ? r : r-step;
    *x = u-v; *y = u+v;
}
void wrap_tile_coords(Map *map, int *x, int *y) {
    if (map->Flags & 1) *x = (*x % map->Width + map->Width) % map->Width;
    if (map->Flags & 2) *y = (*y % map->Height + map->Height) % map->Height;
}
bool Map_in_range(Map *map, int unused, int x, int y) {
    (void)unused; return x >= 0 && y >= 0 && x < map->Width && y < map->Height;
}
Tile *tile_at(int x, int y) {
    ++raw_reads;
    if (!Map_in_range(&bic.Map, 0, x, y) || ((x + y) & 1)) return p_null_tile;
    return native_tile_ptrs[(y * bic.Map.Width + x) / 2];
}
unsigned capture_custom_renderer_visibility(Tile *tile, int viewer, int x, int y) {
    assert(tile != NULL && tile != p_null_tile && viewer == 1);
    assert(Map_in_range(&bic.Map, 0, x, y));
    return C3X_RENDERER_TILE_VISIBILITY_KNOWN |
        (explored ? C3X_RENDERER_TILE_EXPLORED : 0) |
        (visible ? C3X_RENDERER_TILE_VISIBLE : 0);
}
bool has_active_building(City *value, int improvement) {
    assert(value == &city && improvement == 0); ++wall_reads; return wall_active;
}
bool is_online_game(void) { return online; }
unsigned current_thread_id(void) { return thread_id; }
int world_change(int x, int y) {
    ++changed_locations; last_changed_x = x; last_changed_y = y; return world_result;
}
int world_move(int x, int y, int new_x, int new_y) {
    assert(x == city.Body.X && y == city.Body.Y && x == new_x && y == new_y);
    ++sight_moves; return world_result;
}
int world_reconcile(void) { ++reconciliations; return reconcile_result; }
void recompute_distribution_hub_totals(void) { ++distribution_refreshes; }
void City_recompute_yields_and_happiness(City *value) {
    assert(value == &city); ++native_yields; ++value->Body.FoodIncome;
}
void City_update_culture(City *value) {
    assert(value == &city); ++native_culture;
    value->Body.CultureIncome = 3;
    value->Body.Total_Cultures[value->Body.CivID] += 3;
    if (native_level >= 0) value->Body.cultural_level = native_level;
}
void calculate_district_culture_science_bonuses(City *value, int *culture, int *science) {
    assert(value == &city && culture != NULL && science == NULL); *culture = culture_bonus;
}
void City_recompute_cultural_level(City *value, int unused, char a, char b, char c) {
    assert(value == &city && unused == 0 && a == 0 && b == 0 && c == 0);
    ++cultural_recomputes;
    if (final_level >= 0) value->Body.cultural_level = final_level;
}
bool read_custom_renderer_world_record(struct c3x_renderer_tile_v1 *record,
        int viewer, int mask, int x, int y, Tile *tile) {
    assert(viewer == 1 && mask == 77 && tile == native_tile_ptrs[(y * bic.Map.Width + x) / 2]);
    if (read_failure_at >= 0 && copied_records == read_failure_at) return false;
    if (is->custom_renderer_initial_world_capture) ++initial_appearance_reads; else ++hidden_delta_reads;
    ++copied_records;
    *record = (struct c3x_renderer_tile_v1){0};
    record->tile_x = x; record->tile_y = y;
    record->tile_flags = C3X_RENDERER_TILE_VISIBILITY_KNOWN | C3X_RENDERER_TILE_TOPOLOGY_HALO;
    if (visible) record->tile_flags |= C3X_RENDERER_TILE_PREFETCH;
    return true;
}
void log_custom_renderer_event(char const *event, int result) {
    assert(strcmp(event, "initial-world-seed") == 0); ++log_calls; logged_result = result;
}
int capture_custom_renderer_world_page(struct c3x_renderer_world_page_v1 *page);
int seed_world(struct c3x_renderer_camera_request_v1 const *value) {
    assert(value == &request && is->custom_renderer_initial_world_capture);
    ++seed_calls;
    for (unsigned first = 0; first < 320; first += 128) {
        struct c3x_renderer_world_page_v1 page = {0};
        page.struct_size = sizeof page; page.first = first; page.capacity = 128;
        page.identity = value->identity; page.frame = *value->frame; page.tiles = output;
        int result = capture_custom_renderer_world_page(&page);
        if (result != C3X_RENDERER_RESULT_OK) return result;
        ++seed_pages;
        assert(page.count == (first == 256 ? 64u : 128u));
        for (unsigned n = 0; n < page.count; ++n) {
            unsigned index = first + n;
            assert(output[n].tile_y == (int)(index / 16));
            assert(output[n].tile_x == (int)(2 * (index % 16) + ((index / 16) & 1)));
        }
    }
    return seed_result;
}
'''


CASES = r'''
void clear_events(void) {
    wall_reads = raw_reads = copied_records = changed_locations = reconciliations = sight_moves = 0;
    native_yields = native_culture = distribution_refreshes = cultural_recomputes = 0;
    seed_calls = seed_pages = log_calls = 0; logged_result = -1;
    hidden_delta_reads = initial_appearance_reads = 0;
    state.custom_renderer_world_audit_needed = false;
    state.custom_renderer_redraw_pending = false;
    state.custom_renderer_dirty_flags = 0;
    memset(animator_words, 0, sizeof animator_words);
}
void reset(void) {
    memset(&state, 0, sizeof state); memset(&bic, 0, sizeof bic);
    memset(&screen, 0, sizeof screen); memset(leaders, 0, sizeof leaders);
    memset(&city, 0, sizeof city); memset(captured, 0, sizeof captured);
    p_main_screen_form = &screen;
    bic.Map.Width = 32; bic.Map.Height = 20; bic.Map.Tiles = native_tile_ptrs;
    bic.General.MaximumSize_Town = 6; bic.General.MaximumSize_City = 12;
    bic.Races = races; bic.Eras = eras; bic.Improvements = improvements;
    bic.RacesCount = bic.ErasCount = bic.ImprovementsCount = 2;
    races[0].CultureGroupID = 4; strcpy(races[0].CountryName, "Country");
    strcpy(races[0].SingularName, "Civilization"); strcpy(eras[0].Name.S, "Era");
    improvements[0].Combat_Bombard = 10; improvements[1].Combat_Bombard = 0;
    leaders[1].CapitalID = 42;
    city.Body.ID = 42; city.Body.X = city.Body.Y = 8; city.Body.CivID = 1;
    city.Body.Population.Size = 3; city.Body.cultural_level = 1;
    screen.Player_CivID = 1; screen.TileX_Max = screen.TileY_Max = 12;
    screen.animator.field_18E4 = animator_words;
    state.current_config.enable_custom_rendering = true;
    state.custom_renderer_init_state = IS_OK;
    state.custom_renderer_viewer_civ_id = 1; state.custom_renderer_viewer_epoch = 2;
    state.custom_renderer_map_epoch = 3; state.custom_renderer_visibility_revision = 4;
    state.custom_renderer_world_topology_revision = 5;
    state.custom_renderer_tiles = captured; state.custom_renderer_tile_count = 1;
    state.custom_renderer_world_topology = topology; state.custom_renderer_world_topology_count = 320;
    state.custom_renderer_world_visibility = visibility_values;
    state.custom_renderer_world_change = world_change;
    state.custom_renderer_world_move = world_move;
    state.custom_renderer_world_reconcile = world_reconcile;
    state.custom_renderer_seed_world = seed_world;
    state.custom_renderer_probe_thread_id = current_thread_id;
    state.custom_renderer_probe_owner = thread_id = 7;
    state.custom_renderer_native_probe_active = true;
    state.custom_renderer_native_observe = &city;
    state.custom_renderer_capture_world_topology = true;
    state.custom_renderer_display_valid = true;
    state.custom_renderer_target = &bic.Map.Renderer;
    visible = explored = true; wall_active = online = false; debug_bits = 0;
    native_level = final_level = read_failure_at = -1; culture_bonus = 0;
    world_result = reconcile_result = seed_result = C3X_RENDERER_RESULT_OK;
    for (unsigned n = 0; n < 320; ++n) {
        native_tile_ptrs[n] = &native_tiles[n];
        memset(&native_tiles[n], 0, sizeof native_tiles[n]);
        visibility_values[n] = 0;
    }
    capture_custom_renderer_city_body(&captured[0], &city);
    captured[0].tile_x = city.Body.X; captured[0].tile_y = city.Body.Y;
    captured[0].tile_flags = C3X_RENDERER_TILE_VISIBILITY_KNOWN |
        C3X_RENDERER_TILE_EXPLORED | C3X_RENDERER_TILE_VISIBLE;
    captured[0].visibility_mask = 77;
    frame = (struct c3x_renderer_frame_v1){0};
    frame.tiles = captured; frame.world_topology = topology; frame.world_topology_count = 320;
    frame.world_width_tiles = 32; frame.world_height_tiles = 20;
    frame.world_topology_revision = 5;
    request = (struct c3x_renderer_camera_request_v1){0};
    request.version = C3X_RENDERER_CAMERA_VIEW_VERSION; request.struct_size = sizeof request;
    request.frame = &frame;
    request.identity = (struct c3x_renderer_camera_identity_v1){3, 2, 4, 5};
    clear_events();
}
void no_redraw(void) {
    assert(!state.custom_renderer_redraw_pending && state.custom_renderer_dirty_flags == 0);
    assert(animator_words[10] == 0);
}
void city_body_case(void) {
    reset();
    patch_City_recompute_yields_and_happiness(&city);
    assert(native_yields == 1 && changed_locations == 0 && reconciliations == 0);
    assert(!state.custom_renderer_world_audit_needed); no_redraw();
    city.Body.Population.Size = 6; // Label change remains in the town body class.
    patch_City_recompute_yields_and_happiness(&city);
    assert(changed_locations == 0 && native_yields == 2); no_redraw();
    clear_events(); city.Body.Population.Size = 7; // Mutation already happened before hook entry.
    patch_City_recompute_yields_and_happiness(&city);
    assert(native_yields == 1 && changed_locations == 1);
    assert(last_changed_x == 8 && last_changed_y == 8);
    assert(state.custom_renderer_redraw_pending && state.custom_renderer_dirty_flags == C3X_RENDERER_DIRTY_SCENE);
    assert(*(bool *)(animator_words + 10) && !state.custom_renderer_world_audit_needed);
    reset(); wall_active = true;
    notify_custom_renderer_city_change(&city, -1);
    assert(wall_reads == 1 && changed_locations == 1 && state.custom_renderer_redraw_pending);
    reset(); city.Body.X = 24; city.Body.Y = 16;
    captured[0].tile_x = 24; captured[0].tile_y = 16; city.Body.Population.Size = 7;
    notify_custom_renderer_city_change(&city, -1);
    assert(changed_locations == 1 && last_changed_x == 24 && last_changed_y == 16); no_redraw();
    reset(); visible = false; wall_active = true; city.Body.Population.Size = 7;
    captured[0].tile_flags &= ~C3X_RENDERER_TILE_VISIBLE;
    notify_custom_renderer_city_change(&city, -1);
    assert(wall_reads == 0 && changed_locations == 1 && reconciliations == 0 && sight_moves == 0);
    assert(state.custom_renderer_redraw_pending);
    reset(); visible = false; wall_active = true; city.Body.Population.Size = 7;
    notify_custom_renderer_city_change(&city, -1);
    assert(sight_moves == 1 && changed_locations == 0 && wall_reads == 0);
    assert(state.custom_renderer_redraw_pending && !state.custom_renderer_world_audit_needed);
    reset(); visible = explored = false; wall_active = true; city.Body.Population.Size = 7;
    captured[0].tile_flags &= ~(C3X_RENDERER_TILE_VISIBLE | C3X_RENDERER_TILE_EXPLORED);
    notify_custom_renderer_city_change(&city, -1);
    assert(wall_reads == 0 && changed_locations == 0 && sight_moves == 0); no_redraw();
    reset(); city.Body.Population.Size = 7; world_result = C3X_RENDERER_RESULT_PENDING;
    notify_custom_renderer_city_change(&city, -1);
    assert(changed_locations == 1 && state.custom_renderer_world_audit_needed);
}
void city_config_case(void) {
    reset(); state.current_config.enable_custom_rendering = false;
    state.current_config.enable_districts = true;
    state.current_config.enable_distribution_hub_districts = true;
    state.distribution_hub_totals_dirty = true;
    city.Body.Population.Size = 7;
    patch_City_recompute_yields_and_happiness(&city);
    assert(native_yields == 1 && distribution_refreshes == 1 && raw_reads == 0 && wall_reads == 0);
    assert(changed_locations == 0 && reconciliations == 0); no_redraw();
    culture_bonus = 9; final_level = 2;
    patch_City_update_culture(&city);
    assert(native_culture == 1 && cultural_recomputes == 1);
    assert(city.Body.CultureIncome == 12 && city.Body.Total_Cultures[1] == 12);
    assert(city.Body.cultural_level == 2 && changed_locations == 0 && reconciliations == 0); no_redraw();
    reset(); state.current_config.enable_custom_rendering = false;
    state.current_config.enable_natural_wonders = true; culture_bonus = -9;
    patch_City_update_culture(&city);
    assert(native_culture == 1 && cultural_recomputes == 1);
    assert(city.Body.CultureIncome == 0 && city.Body.Total_Cultures[1] == 0); no_redraw();
}
void city_culture_case(void) {
    reset(); patch_City_update_culture(&city);
    assert(native_culture == 1 && reconciliations == 0 && changed_locations == 0);
    assert(!state.custom_renderer_world_audit_needed); no_redraw();
    reset(); native_level = 2; patch_City_update_culture(&city);
    assert(reconciliations == 1 && changed_locations == 0 && !state.custom_renderer_world_audit_needed); no_redraw();
    reset(); state.current_config.enable_districts = true; culture_bonus = 17; final_level = 3;
    patch_City_update_culture(&city);
    assert(city.Body.cultural_level == 3 && cultural_recomputes == 1 && reconciliations == 1);
    assert(changed_locations == 0 && !state.custom_renderer_world_audit_needed); no_redraw();
    reset(); visible = false; native_level = 2;
    captured[0].tile_flags &= ~C3X_RENDERER_TILE_VISIBLE;
    patch_City_update_culture(&city);
    assert(reconciliations == 1 && wall_reads == 1 && changed_locations == 0); no_redraw();
    reset(); native_level = 2; reconcile_result = C3X_RENDERER_RESULT_PENDING;
    patch_City_update_culture(&city);
    assert(reconciliations == 1 && state.custom_renderer_world_audit_needed); no_redraw();
}
void city_culture_view_case(void) {
    for (int shrink = 0; shrink < 2; ++shrink) {
        reset();
        city.Body.X = captured[0].tile_x = 24;
        city.Body.Y = captured[0].tile_y = 16;
        assert(!custom_renderer_tile_near_view(city.Body.X, city.Body.Y, 6));
        city.Body.cultural_level = shrink ? 3 : 1;
        native_level = shrink ? 1 : 2;
        state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch;
        // Synchronous completed maps need not have the asynchronous display flag.
        state.custom_renderer_display_valid = false;
        patch_City_update_culture(&city);
        assert(native_culture == 1 && city.Body.cultural_level == native_level);
        assert(reconciliations == 1 && changed_locations == 0 && sight_moves == 0);
        assert(!state.custom_renderer_world_audit_needed);
        assert(state.custom_renderer_redraw_pending);
        assert(state.custom_renderer_dirty_flags == C3X_RENDERER_DIRTY_SCENE);
        assert(*(bool *)(animator_words + 10));
    }
    reset(); state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch;
    patch_City_recompute_yields_and_happiness(&city);
    patch_City_update_culture(&city);
    assert(native_yields == 1 && native_culture == 1);
    assert(reconciliations == 0 && changed_locations == 0); no_redraw();

    reset(); state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch;
    state.current_config.enable_custom_rendering = false; native_level = 2;
    patch_City_update_culture(&city);
    assert(native_culture == 1 && city.Body.cultural_level == 2);
    assert(reconciliations == 0 && changed_locations == 0 && raw_reads == 0 && wall_reads == 0);
    no_redraw();

    for (int display_epoch = 0; display_epoch < 4; ++display_epoch) {
        if (display_epoch == state.custom_renderer_viewer_epoch) continue;
        reset(); state.custom_renderer_display_viewer_epoch = display_epoch;
        native_level = 2; patch_City_update_culture(&city);
        assert(native_culture == 1 && reconciliations == 1 && changed_locations == 0);
        no_redraw();
    }
    reset(); state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch;
    screen.animator.field_18E4 = NULL;
    native_level = 2; patch_City_update_culture(&city);
    assert(reconciliations == 1 && state.custom_renderer_redraw_pending);
    assert(state.custom_renderer_dirty_flags == C3X_RENDERER_DIRTY_SCENE);
    assert(animator_words[10] == 0);
}
struct c3x_renderer_world_page_v1 page(void) {
    struct c3x_renderer_world_page_v1 value = {0};
    value.struct_size = sizeof value; value.capacity = 128; value.tiles = output;
    value.identity = request.identity; value.frame = frame; return value;
}
void initial_scope(void) {
    state.custom_renderer_initial_world_capture = true;
    state.custom_renderer_draw_in_progress = state.custom_renderer_frame_active = true;
    state.custom_renderer_display_valid = false; screen.is_now_loading_game = true;
}
void world_ordinary_case(void) {
    reset(); struct c3x_renderer_world_page_v1 value = page();
    assert(capture_custom_renderer_world_page(&value) == C3X_RENDERER_RESULT_OK);
    assert(value.count == 128 && raw_reads == 128 && copied_records == 128);
    assert(hidden_delta_reads == 128 && initial_appearance_reads == 0);
    for (int condition = 0; condition < 8; ++condition) {
        reset(); value = page();
        switch (condition) {
        case 0: screen.is_now_loading_game = true; break;
        case 1: state.custom_renderer_display_valid = false; break;
        case 2: state.custom_renderer_draw_in_progress = true; break;
        case 3: state.custom_renderer_frame_active = true; break;
        case 4: state.custom_renderer_capture_only = true; break;
        case 5: screen.Player_CivID = 2; break;
        case 6: state.custom_renderer_init_state = IS_INIT_FAILED; break;
        case 7: state.current_config.enable_custom_rendering = false; break;
        }
        assert(capture_custom_renderer_world_page(&value) == C3X_RENDERER_RESULT_PENDING);
        assert(raw_reads == 0 && copied_records == 0);
    }
    reset(); value = page(); value.first = 0xffffffffu; value.count = 1;
    output[0].tile_x = output[0].tile_y = 8;
    assert(capture_custom_renderer_world_page(&value) == C3X_RENDERER_RESULT_OK);
    assert(value.count == 0 && copied_records == 0); // Ordinary unchanged visibility delta.
    value.first = 0xfffffffeu; value.count = 1;
    assert(capture_custom_renderer_world_page(&value) == C3X_RENDERER_RESULT_OK);
    assert(value.count == 1 && copied_records == 1); // Ordinary force-local body delta.
}
void world_initial_case(void) {
    for (int condition = 0; condition < 30; ++condition) {
        reset(); initial_scope(); struct c3x_renderer_world_page_v1 value = page();
        int expected = C3X_RENDERER_RESULT_PENDING;
        switch (condition) {
        case 0: expected = C3X_RENDERER_RESULT_OK; break;
        case 1: thread_id = 8; break;
        case 2: state.custom_renderer_probe_thread_id = NULL; break;
        case 3: state.custom_renderer_capture_failed = true; break;
        case 4: state.custom_renderer_capture_only = true; break;
        case 5: state.custom_renderer_target = NULL; break;
        case 6: state.custom_renderer_tile_count = 0; break;
        case 7: state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch; break;
        case 8: bic.Map.Tiles = NULL; break;
        case 9: bic.Map.Width = 0; break;
        case 10: bic.Map.Height = 0; break;
        case 11: bic.Map.Width = 31; break;
        case 12: bic.Map.Height = 2049; break;
        case 13: state.custom_renderer_world_topology = NULL; break;
        case 14: state.custom_renderer_world_topology_count = 319; break;
        case 15: ++value.identity.map_epoch; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 16: ++value.identity.viewer_epoch; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 17: ++value.identity.visibility_epoch; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 18: ++value.identity.scene_epoch; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 19: ++value.frame.world_width_tiles; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 20: value.frame.world_wrap_x = 1; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 21: ++value.frame.world_topology_revision; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 22: value.first = 320; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 23: value.first = 0xffffffffu; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 24: value.first = 0xfffffffeu; expected = C3X_RENDERER_RESULT_SUPERSEDED; break;
        case 25: value.capacity = 129; expected = C3X_RENDERER_RESULT_BAD_ARGUMENT; break;
        case 26: value.capacity = 0; expected = C3X_RENDERER_RESULT_BAD_ARGUMENT; break;
        case 27: value.tiles = NULL; expected = C3X_RENDERER_RESULT_BAD_ARGUMENT; break;
        case 28: --value.struct_size; expected = C3X_RENDERER_RESULT_BAD_ARGUMENT; break;
        case 29: p_main_screen_form = NULL; break;
        }
        int result = capture_custom_renderer_world_page(&value);
        if (result != expected) fprintf(stderr, "Initial callback case %d: %d != %d\n", condition, result, expected);
        assert(result == expected);
        assert(raw_reads == (condition == 0 ? 128 : 0));
        assert(copied_records == (condition == 0 ? 128 : 0));
        assert(initial_appearance_reads == (condition == 0 ? 128 : 0) && hidden_delta_reads == 0);
    }
    reset(); initial_scope(); assert(capture_custom_renderer_world_page(NULL) == C3X_RENDERER_RESULT_BAD_ARGUMENT);
    assert(raw_reads == 0);
}
void seed_case(void) {
    reset(); state.custom_renderer_draw_in_progress = state.custom_renderer_frame_active = true;
    screen.is_now_loading_game = true; state.custom_renderer_display_valid = false;
    assert(seed_custom_renderer_initial_world(&request) == C3X_RENDERER_RESULT_OK);
    assert(seed_calls == 1 && seed_pages == 3 && copied_records == 320);
    assert(initial_appearance_reads == 320 && hidden_delta_reads == 0);
    assert(!state.custom_renderer_initial_world_capture && log_calls == 1 && logged_result == C3X_RENDERER_RESULT_OK);
    assert(state.custom_renderer_seeded_viewer_epoch == state.custom_renderer_viewer_epoch);
    assert(seed_custom_renderer_initial_world(&request) == C3X_RENDERER_RESULT_OK);
    assert(seed_custom_renderer_initial_world(&request) == C3X_RENDERER_RESULT_OK);
    assert(seed_calls == 1 && seed_pages == 3 && copied_records == 320);
    for (int condition = 0; condition < 18; ++condition) {
        reset(); state.custom_renderer_draw_in_progress = state.custom_renderer_frame_active = true;
        switch (condition) {
        case 0: state.current_config.enable_custom_rendering = false; break;
        case 1: state.custom_renderer_capture_world_topology = false; break;
        case 2: state.custom_renderer_seed_world = NULL; break;
        case 3: state.custom_renderer_native_probe_active = false; break;
        case 4: thread_id = 8; break;
        case 5: state.custom_renderer_draw_in_progress = false; break;
        case 6: state.custom_renderer_frame_active = false; break;
        case 7: state.custom_renderer_capture_only = true; break;
        case 8: state.custom_renderer_capture_failed = true; break;
        case 9: state.custom_renderer_target = NULL; break;
        case 10: state.custom_renderer_tile_count = 0; break;
        case 11: bic.Map.Tiles = NULL; break;
        case 12: state.custom_renderer_viewer_civ_id = -1; break;
        case 13: request.version = 99; break;
        case 14: request.frame = NULL; break;
        case 15: frame.world_topology = NULL; break;
        case 16: ++request.identity.viewer_epoch; break;
        case 17: ++request.identity.scene_epoch; break;
        }
        assert(seed_custom_renderer_initial_world(&request) == C3X_RENDERER_RESULT_BAD_ARGUMENT);
        assert(seed_calls == 0 && raw_reads == 0 && !state.custom_renderer_initial_world_capture);
    }
    reset(); state.custom_renderer_draw_in_progress = state.custom_renderer_frame_active = true;
    seed_result = C3X_RENDERER_RESULT_ERROR;
    assert(seed_custom_renderer_initial_world(&request) == C3X_RENDERER_RESULT_ERROR);
    assert(seed_calls == 1 && !state.custom_renderer_initial_world_capture && logged_result == C3X_RENDERER_RESULT_ERROR);
    assert(state.custom_renderer_seeded_viewer_epoch == 0);
    clear_events(); seed_result = C3X_RENDERER_RESULT_OK; read_failure_at = 11;
    assert(seed_custom_renderer_initial_world(&request) == C3X_RENDERER_RESULT_ERROR);
    assert(seed_calls == 1 && copied_records == 11 && !state.custom_renderer_initial_world_capture);
    assert(logged_result == C3X_RENDERER_RESULT_ERROR);
    reset(); state.custom_renderer_display_viewer_epoch = state.custom_renderer_viewer_epoch;
    assert(seed_custom_renderer_initial_world(&request) == C3X_RENDERER_RESULT_OK);
    assert(seed_calls == 0 && raw_reads == 0 && !state.custom_renderer_initial_world_capture);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    if (strcmp(argv[1], "city_body") == 0) city_body_case();
    else if (strcmp(argv[1], "city_config") == 0) city_config_case();
    else if (strcmp(argv[1], "city_culture") == 0) city_culture_case();
    else if (strcmp(argv[1], "city_culture_view") == 0) city_culture_view_case();
    else if (strcmp(argv[1], "world_ordinary") == 0) world_ordinary_case();
    else if (strcmp(argv[1], "world_initial") == 0) world_initial_case();
    else if (strcmp(argv[1], "seed") == 0) seed_case();
    else assert(!"Unknown contract case");
    return 0;
}
'''


class InjectedWorldLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("Host clang is unavailable")
        cls.temporary = tempfile.TemporaryDirectory(prefix="c3x-world-local-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        source = (ROOT / "injected_code.c").read_text()
        program = PRELUDE + "\n".join(production_function(source, name) for name in FUNCTIONS) + CASES
        contract = directory / "contract.c"
        contract.write_text(program)
        cls.executable = directory / "contract"
        result = subprocess.run(
            [compiler, "-std=c17", "-O1", "-Wall", "-Wextra", "-Werror",
             "-I", str(ROOT), str(contract), "-o", str(cls.executable)],
            text=True, capture_output=True, timeout=30,
        )
        if result.returncode:
            raise AssertionError("Host C contract compile failed:\n" + result.stdout + result.stderr)

    def contract(self, name):
        result = subprocess.run([str(self.executable), name], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_city_uses_copied_body_visibility_and_local_redraw(self):
        self.contract("city_body")

    def test_config_off_preserves_existing_gameplay_hooks(self):
        self.contract("city_config")

    def test_culture_reconciles_only_after_level_changes(self):
        self.contract("city_culture")

    def test_culture_transition_refreshes_only_a_certified_view(self):
        self.contract("city_culture_view")

    def test_ordinary_world_callback_keeps_loading_and_display_guards(self):
        self.contract("world_ordinary")

    def test_initial_callback_has_bounded_validated_page_scope(self):
        self.contract("world_initial")

    def test_seed_scope_clears_after_success_and_failures(self):
        self.contract("seed")


if __name__ == "__main__":
    unittest.main()
