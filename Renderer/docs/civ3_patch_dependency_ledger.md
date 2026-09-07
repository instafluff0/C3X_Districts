# Civ III Patch Dependency Ledger

## Purpose

This ledger is the renderer project's standing record of Civ III functions needed for capture, scheduling, ownership suppression, animation synchronization, and compositing. It tells the human maintainer when a new `civ_prog_objects.csv` entry is actually required.

Agents must update this document and `project_status.json` as soon as a requirement becomes concrete. They must also report the request directly to the user in the same turn. Candidate functions remain audit items until an existing hook has been ruled out and the required patch form, signature, and supported-executable addresses are known.

Agents do not edit `civ_prog_objects.csv` or `ref/Civ3Conquests.h`.

## Current Human Action

The user-authorized 2026-09-06 production performance maintenance needs no new
CSV entries. It uses the existing `Map_Renderer_m71_Draw_Tiles` lifecycle,
`Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` capture/composite boundary,
`QueryPerformanceCounter`, and `OutputDebugStringA`. Native topology capture now
also calls the existing `Main_Screen_Form_tile_to_screen_coords` definition:
`void (__fastcall *)(Main_Screen_Form *, int edx, int tile_x, int tile_y, int *, int *)`,
at GOG `0x4E3B10`, Steam `0x4EC360`, PCGames `0x4E3BD0`. It is an existing callable,
not a new patch. Every captured logical anchor must agree before adding a halo;
on mismatch the original capture and dependency validation remain the fallback.
The twelve-coordinate topology halo now includes a four-coordinate inner ring
with complete authoritative appearance metadata for optional idle preparation.
Capture stays on the game thread. The single D3D worker prepares immutable tile
meshes from its own snapshot, prioritizes foreground requests, and never modifies
published pixels or ownership arrays during idle work. Cache compilation,
dependency validation, changed-set bitmap damage and diagnostics remain DLL
responsibilities. River-node dependencies use a conservative local window.
The worker also prepares bounded 128-pixel composite blocks using the same
off-screen draw path. Per-map OutputDebugStringA diagnostics include halo capture,
pixel-cache use and whole-map timings. No additional native callable is needed.
`audit_candidates = []` and
`required_user_action = []` for this increment. The API 13 header is shared by
the DLL and injected compilation; the approved smoke must verify both together.
The deferred M7.5 requests below remain unchanged.

**M7.5 needs two new nuclear-outcome inleads and one existing loader upgraded
from `define` to `inlead`.** Exact requests are recorded below. They do not
authorize early M7.5 runtime implementation; until its preceding gates pass,
the nuclear and transient-effect presentation remains native.

M5.1 through M5.3 were implemented with existing symbols. M5.3 proved that the existing timer/map hooks and main-screen animator dirty bit are sufficient; it needs no new callable redraw entry. Other future M6/M7 candidates below are not requests to edit `civ_prog_objects.csv`.

## Status Vocabulary

- `available_patch`: already present in `civ_prog_objects.csv` in a patchable form used by C3X.
- `available_call`: already present as a callable definition, but not necessarily patchable at function entry.
- `audit_candidate`: may be useful, but no human action should be taken until the milestone proves it is necessary.
- `required_user_action`: a new or changed CSV entry is blocking implementation; the exact request must be listed and reported immediately.
- `not_required`: investigated and avoided through an existing boundary or data source.

## Existing Renderer-Relevant Symbols

| Milestone use | Symbol | CSV capability | Current purpose |
|---|---|---|---|
| M5.1-M6.0 | `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` | `available_patch` (`repl vptr`) | Authoritative visible tile coordinates, anchors, pass flags, and terrain metadata. |
| M5.1-M7.3 | `Map_Renderer_m71_Draw_Tiles` | `available_patch` (`repl vptr`) | Bounded retained-map lifecycle for terrain and tile-bound objects; not the later unit-animation plane. |
| M6.0-M7.2 | `Sprite_draw_on_map` | `available_patch` (`inlead`) | Correlate selected sprite pointers/indices with current tile context and runtime census evidence. |
| M6.0-M7.2 | `Map_Renderer_m08_Draw_Tile_Forests_Jungle_Swamp` | `available_patch` (`repl vptr`) | Feature ownership/census. |
| M6.0-M7.2 | `Map_Renderer_m09_Draw_Tile_Resources` | `available_patch` (`repl vptr`) | Resource ownership/census. |
| M6.0-M7.2 | `Map_Renderer_m11_Draw_Tile_Irrigation` | `available_patch` (`inlead`) | Irrigation ownership/census. |
| M6.0-M7.2 | `Map_Renderer_m12_Draw_Tile_Buildings` | `available_patch` (`repl vptr`) | Tile-building ownership/census. |
| M6.0-M7.1 | `Map_Renderer_m52_Draw_Roads` / `Map_Renderer_m52_Draw_Railroads` | `available_patch` (`repl vptr`) | Road/rail index and ownership capture. |
| M5.3 | `on_timer_0x9F6500` | `available_patch` (`inlead`) | Existing approximately 66 ms callback; may mark animation dirty and request an audited normal redraw, never render directly. |
| M5.3-M7.5 | `Animator_update` | `available_call` (`define`) | Actual camera/full-map refresh and later unit/effect dirty-region routine; available to call/inspect, but entry interception is not currently available. |
| M7.4 | `Animator_play_one_shot_unit_animation` | `available_patch` (`inlead`) | Capture one-shot action starts including death, fortify, victory, build, and attack-family calls. |
| M7.4 | `Unit_move` / `Unit_move_to_adjacent_tile` | `available_patch` (`inlead`) | Authoritative movement intent/endpoints and adjacent direction. |
| M7.4 | `Unit_play_attack_animation` | `available_call` (`define`) | Attack animation helper; call interception is not currently available at entry. |
| M7.4 | `Main_Screen_Form_set_selected_unit` | `available_patch` (`inlead`) | Current/selected unit transitions and selected-underlay ownership. |
| M7.4-M7.5 | `Fighter_begin`, `Fighter_animate_start_of_combat`, `Fighter_fight` | `available_patch` (`inlead`) | Combat event identity, participants, direction, and outcome boundaries. |
| M7.4-M7.5 | `Unit_bombard_tile`, `Unit_play_bombard_fire_animation`, `Unit_play_bombing_animation` | mixed existing patch/call entries | Bombard source/target and animation-event boundaries. |
| M7.5 | `Fighter_do_bombard_tile` | `available_patch` (`inlead`) | Wrap authoritative bombard resolution and capture before/after outcome state without changing rolls. |
| M7.5 | `Units_Image_Data_load_animated_effect` | `available_patch` (`inlead`) | Observe exact `AE_Hit*`, `AE_Miss`, and `AE_WaterMiss` impact creation time while retaining native animation/audio state. |
| M7.4-M7.5 | `Unit_despawn` | `available_patch` (`inlead`) | Authoritative unit removal and stale animation cleanup. |

`QueryPerformanceCounter` and `QueryPerformanceFrequency` are already imported into C3X state and do not require `civ_prog_objects.csv` entries.

## Audit Candidates, Not Yet Requests

### M5.3 Frame Scheduling: Resolved, No New Entry

- `patch_on_timer_0x9F6500` marks the existing main-screen animator dirty, then the original callback reaches `Animator::refresh`. The current terrain renderer still enters only through `Map_Renderer_m71_Draw_Tiles`; M7.4 requires a separately proven Animator-plane unit-body boundary. No new CSV function is required for the completed scheduler slice.
- `Animator::refresh` entry interception and a separate redraw call were both unnecessary. They remain ordinary Civ III behavior rather than new renderer dependencies.

### M6.0 Inventory

- A dedicated river, city, fog, or final-overlay draw method only if `m19` pass context plus `Sprite_draw_on_map` cannot correlate its selected art and index reliably.
- BIQ loader/accessor functions only if existing loaded `BIC` structures cannot supply the semantic inventory safely.

### M6.2 Native Normalized Grassland: Resolved, No New Entry

- The existing `Map_Renderer_m71_Draw_Tiles` and `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` bridge already supplies terrain identity and authoritative anchors. M6.2 adds only the DLL-local `c3x_renderer_set_pack_path` export; it is not a Civ III patch symbol and needs no `civ_prog_objects.csv` entry.
- M6.3 continued with the existing capture. Its DLL-local definition-path export, multi-material selection, and per-item fallback required no new Civ III symbol or address.

### M6.3 Definition-Driven Terrain: Resolved, No New Entry

- Existing terrain identity, `SquareParts`, authoritative anchors, hour/season, and the audited compositing boundary were sufficient for the completed definition-driven material slice.
- M6.4 begins with the already captured environment state and M5.3 scheduler. A new attachment selector remains only an audit candidate until a supported effect proves the current scene fields insufficient.

### M6.4 Shared Environment: Resolved, No New Entry

- The native shared environment consumes the existing captured hour, season, and absolute presentation timestamp.
- Static emissives remain idle and the synthetic animated attachment uses the existing M5.3 scheduler contract.
- No new Civ III symbol, address, injected-state field, or `civ_prog_objects.csv` entry was required. Later object categories must prove any additional selector at their own gate.

### M6.5 Connected Base Materials: Resolved, No New Entry

- Existing terrain coordinates and authoritative anchors supply all four valid staggered-map neighbors needed by the DLL-local material blend.
- The existing `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` capture already retains the complete sequence of native companion base calls. Replaying that sequence beneath a custom frame containing any fallback fixes black wedges and supplies the underlay used by partial-alpha mapped/fallback edges.
- No new Civ III symbol, address, injected-state field, or `civ_prog_objects.csv` entry was required.

### I9-I13A Render-Loop Viability And Exclusive Ownership: Resolved, No New Entry

- The decompiled `m71 -> m20 -> m21 -> m19` chain confirms the existing terrain insertion point remains valid across normal camera redraws, both zoom bases, clipped traversal, and wrapped logical coordinates with distinct screen occurrences.
- Existing `load_scenario`, `Map_Renderer_m71_Draw_Tiles`, and DLL reset exports are sufficient to re-resolve scenario definitions, exclude nested draws, and rebuild after a D3D device error on a later normal frame.
- Cache reuse must compare a content-derived frame/fragment fingerprint independent of timer dirty hints; this is a DLL/API implementation contract, not a new Civ III patch dependency.
- The user made custom-on ownership exclusive. The existing `m19` hook captures and composites at the audited boundary but never calls or replays the original multiplexed tile renderer while enabled; loader, capture, render, validation, blit, device, and reentrant failures do not switch to vanilla. This intentionally hides unported tile layers and requires no new symbol.
- The current production smoke covers both zoom bases, clipped and pixel-scrolled frames, a duplicate horizontal-wrap occurrence, reset recovery, bounded 32-bit live-scale capture, cache telemetry, terrain/feature/river ownership, active/dormant volcano invalidation, authoritative river-mask invalidation, four lighting phases, and zero native fallback with the frozen L9-L13A payloads.
- Existing `m49_Get_Square_RealType`, `m50_Get_Square_BaseType`, `Tile.Body.active_tile_effect`, `Tile::m37_Get_River_Code`, captured hour/season, and the m19 anchor/clip/wrap context provide every I13/I13A selector. The river and shared-lighting work is DLL-local apart from adding a river replacement ownership bit to the existing bridge contract. No new symbol or address is required for I9 through I13A. Automated evidence remains in `approved_terrain_integration.md`, `terrain/i12_handoff_fidelity.json`, `terrain/i13_handoff_fidelity.json`, `terrain/i13a_handoff_fidelity.json`, and the corresponding evidence directories.

### I14-I18 Routes, Resources, Cities, And Mines: Resolved, No New Entry

- The existing `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` capture supplies
  authoritative road/rail topology, tile improvement flags, resource identity,
  and exact anchors at the already exclusive custom map-plane boundary.
- Loaded `BIC` resource records provide visibility-conditioned resource class
  and name. Loaded city/leader/improvement structures provide city identity,
  owner, size band, culture group, era, capital, walls, and the visible
  civilization era used by routes and mines. No separate map-body hook is
  necessary; Civ III city labels, resource UI icons, and all retained overlays
  continue after the custom plane.
- API v9 adds DLL-local route/resource/city/mine selectors and exact ownership
  bits. The 32-entry/128 MiB exact viewport LRU and 96 MiB canonical world-tile
  sample cache include every renderer-owned
  selector but excludes unit animation, selection, native `SquareParts`, fog,
  and exact population values that cannot change this plane.
- One capacity-one renderer worker consumes copied immutable snapshots and owns
  renderer/D3D mutation. Existing m19/m71 capture, reset, and blit boundaries are
  sufficient; no worker-specific Civ III hook or symbol is required.
- The production smoke consumes the frozen approved L14-L18 payloads at both
  zooms and proves route/resource/city/mine invalidation, LRU reuse/eviction,
  exact clips, scrolling/wrap occurrences, deterministic reset, and zero native
  replay. Evidence is in `docs/approved_terrain_integration.md`,
  `evidence/i14_i18/README.md`, and
  `verification/i18_cache_integration.json`.
- `required_user_action: []`. No new or changed `civ_prog_objects.csv` symbol,
  signature, or supported-build address is required for I14 through I18.

### M7.2 Map Resources

- Begin with existing `Map_Renderer_m09_Draw_Tile_Resources`, `Sprite_draw_on_map`, loaded BIQ resource records, and the captured map anchor/index context. Suppress only the map draw call for a fully accepted replacement; never replace or mutate the shared resource PCX/UI icon paths globally.
- Audit a dedicated player-resource-visibility accessor only if the existing draw decision and captured player context cannot reproduce hidden strategic-resource behavior. No new Civ III symbol/address is requested until that insufficiency is demonstrated.

### M7.1 Goody Huts And Colonies: Resolved, No New Entry

- Goody presence/visibility is already available through
  `Tile::m15_Check_Goody_Hut(viewer_civ_id)` during the existing m19 tile
  capture. Map seed and canonical tile index are sufficient for stable variant
  selection; the native eight-way image index is not persistent gameplay state.
- Colony presence and identity are already available through `Tile_has_colony`,
  tile-building ID, `p_colonies`, and `Tile_Building_Body.{X,Y,OwnerID}`. The
  colony owner ID—not tile territory owner—selects leader era and effective
  owner-color row, preserving C3X extraterritorial colonies.
- The existing `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` exclusive plane and
  `Map_Renderer_m12_Draw_Tile_Buildings` census hook are sufficient. Separate
  `m26_Draw_Colony` or `m34_Draw_Goody_Huts` patch entries are not required.
  The checked offline handoff is `docs/goody_huts_and_colonies.md`.

### M7.1 Remaining Tile Infrastructure: Resolved, No New Entry

- The existing m19 capture already records `tile_building_id` plus the
  `C3X_RENDERER_IMPROVEMENT_TILE_BUILDING`, `POLLUTION`, and `CRATER` flags from
  Civ III's visibility-conditioned tile accessors. This covers Fortress,
  Barricade, Airfield, Outpost, Radar Tower, Pollution, Crater, and Victory
  Location identity/state without a new hook.
- The existing `Map_Renderer_m12_Draw_Tile_Buildings` census and exclusive m19
  map-plane boundary remain sufficient for draw ownership and retained-layer
  ordering. L19B/I19B must prove exact tile-building-ID semantics and native
  suppression, but no new symbol/address is requested by the offline intake.
- Source-art status and the dedicated L19B boundary are in
  `docs/remaining_tile_infrastructure.md`.

### M7.3 Cities

- A city-specific draw boundary only if tile/city state captured during `m19` cannot reproduce culture, era, size, walls, capital state, selected sprite, and anchor while preserving labels and HUD.

### M7.4 Units

- The terrain `m19` record is not a unit-render boundary. `Animator::update` builds the visible stack, chooses a wrap-aware origin, computes `Unit::FUN_005cbc30` dirty rectangles, and calls `Unit::tick_anim` on the separate `Units_Control` canvas.
- `Unit::tick_anim` combines the selected-unit underlay, direct FLC body blit, and retained health/activity/stack work. M7.4 must select the smallest supported-build hook that can replace only the direct body blit while preserving this order and Civ III's dirty accounting.
- Existing `Unit_move`, `Unit_move_to_adjacent_tile`, `Fighter_begin`, `Fighter_animate_start_of_combat`, `Fighter_fight`, `Animator_play_one_shot_unit_animation`, `Main_Screen_Form_set_selected_unit`, and `Unit_despawn` entries cover the currently known movement, combat, selection, one-shot, outcome, and removal event boundaries. Native `AnimationSummary` pixel/action/direction/progress fields cover per-update placement and clip state. No new state-director symbol is presently required.
- Owner tint needs no per-scenario baked art and no new symbol: read the effective 64-color rows from `Units_Image_Data.Color_Tables[0..31].JGL_Color_Table` after Civ III loads the game/scenario, then select `Leader.Color_Table_ID` per displayed civilization. The selector must preserve `Unit::tick_anim`'s viewer-conditioned hidden-nationality choice instead of assuming `Unit.Body.CivID` is always the displayed civ. The executable offline contract is `tools/asset_compiler/unit_owner_color_runtime.json`.
- Army composition needs no separate selection patch. Civ III's dedicated body path reads decompiled `field_1B0[1]` as the displayed-member unit ID, advances both the Army and member FLCs, and already includes both in dirty accounting; `Unit_select_army_member_for_combat` is also an existing inlead. The eventual generic body boundary must capture/suppress both bodies atomically while retaining the parent HUD once. The exact contract and 40/20-pixel native offset references are in `army_rendering_strategy.md` and `tools/asset_compiler/army_render_strategy.json`.
- The offline compound-unit compiler consumes only source-pack structures and emits the same generic node/component/socket/clip recipe for mounted, crewed-siege, armored-crew, and mounted-commander examples. It requires no Civ III patch or address. Runtime atomic body replacement still belongs to the future M7.4 body boundary above; no new symbol is requested by this compiler proof.
- The only presently identified likely new dependency is a body-only suppression/insertion boundary. Begin with exact supported-build disassembly of `Unit::tick_anim` and its normal/reduced direct FLC body calls. Likely candidates are a `Unit_tick_anim` inlead and/or exact replacements around `Sprite::FUN_005f88b0` and `Sprite::FUN_005f8940`, but none is a request until the smallest safe combination, signatures, and all three executable addresses are proven.
- The installed GOG audit now confirms `Unit::tick_anim` at `0x5CBF50`, its normal/reduced body calls at `0x5CC33B` and `0x5CC3A1`, and the retained HUD call afterward at `0x5CC41D`. The Army helper at `0x5CC430` calls the same normal body routine at `0x5CC861`/`0x5CC8BC`, the same reduced routine at `0x5CC926`/`0x5CC98D`, then its one retained HUD at `0x5CC9EB`. The preferred candidate is therefore a scoped `Unit_tick_anim` inlead plus guarded inleads for the two shared Sprite routines: unrelated callers pass through, while one preflight decision suppresses an ordinary body or both Army bodies atomically. Steam/PCGames addresses and the reduced-call signature remain unresolved, so `required_user_action: []`; see `i20_unit_body_replacement_spike.md`.
- `Unit::animate_move` for precise visual segment start/end/progress if existing `Unit_move_to_adjacent_tile`, unit animation state, and authoritative pixel positions are insufficient.
- `Animator::refresh` or `Animator::tick_all_unit_anims` for immutable per-frame unit snapshots and vanilla-unit suppression if the existing map/timer boundaries cannot supply them.
- `FLC_Animation::tick`, `Unit::tick_anim`, or the routine currently decompiled as `Units_Image_Data::FUN_00405fc0` only if clip phase and unit-cursor phase cannot be read without interception.
- The exact vanilla unit-body draw/suppression boundary if the eventual unit draw hook cannot omit only the body while preserving the native selection cursor/ring, health bar, left-side activity/status marks, stack indicators, and related unit HUD.
- The exact unit-cursor draw routine only if `Main_Screen_Form_set_selected_unit`, `Animator.Unit_Cursor_Animation`, and the eventual unit draw boundary cannot keep the retained selected-unit underlay anchored and layered correctly. M7.4 does not replace that underlay.
- The complete implementation handoff and rejection criteria are in `i20_native_unit_animation_handoff.md`.

### M7.5 Effects

- The existing `Unit_bombard_tile`, bombard-fire replacement call,
  `Fighter_do_bombard_tile`, and `Units_Image_Data_load_animated_effect` hooks
  are sufficient to prototype stable begin/release/impact/outcome/cleanup
  events. IDs 3--6 are native hit variants, 7 is land miss, and 8 is water
  miss. Fire Rate may generate multiple exact impact calls from one attack.
- Pixel-only suppression is resolved. `Animator::update` ticks each effect FLC
  before testing byte `0x184` and calling the shared sprite blitter. The
  existing `Units_Image_Data_load_animated_effect` inlead can call native load,
  emit/accept a custom event, and clear only `*(byte *)&anim->Last` for owned
  IDs 3--8. Native sound, frame advancement, dirty accounting, 500 ms waits,
  and lifetime remain intact. A new Animator/draw symbol is `not_required`.
- The standalone SDI interception FLC bypasses that helper and calls
  `Units_Image_Data_load_animation`. Its existing callable definition must
  become an inlead so the identical post-load rule can recognize
  `Art\Animations\SDI\SDI.ini`. This is a changed-capability request below.
- The nuclear boundary is resolved. `animate_nuclear_strike` is delivery only;
  `Unit::do_nuke_tile` is authoritative detonation and
  `Unit::get_intercepted_as_nuke` is authoritative interception. The
  multiplayer sync callback reuses both methods. Nuclear victim despawns are
  outcome detail and are not a detonation trigger.
- Full evidence and the fail-closed ownership rules are in
  `bombardment_and_explosion_effects.md`.

## Required User Action: M7.5 Native Boundaries

### Upgrade `Units_Image_Data_load_animation`

- Milestone/step: M7.5 effects and nuclear interception.
- Symbol name: `Units_Image_Data_load_animation`.
- Reason existing hooks are insufficient: ordinary `AnimatedEffect` loads are
  already intercepted, but the audio-bearing SDI animation is a standalone FLC
  loaded directly through this function. A post-load hook is required to clear
  only its draw-enable byte after renderer ownership succeeds.
- Required CSV capability: change `define` to `inlead`.
- C signature: `void (__fastcall *)(Units_Image_Data * this, int edx, char * asset_string, FLC_Animation * anim, int civ_id, int param_4, int param_5, bool param_6)`.
- Supported executable addresses: GOG `0x4062A0`; Steam `0x406810`;
  PCGames `0x4062D0`.
- Call sites or vtable slots: function entry; existing CSV row 972.
- Fallback while missing: leave standalone SDI pixels and the entire custom
  nuclear-interception presentation native.
- Verification unlocked: prove SDI pixels are absent while its native INI/FLC
  sound, animation advancement, wait, and cleanup still occur.

### Add `Unit_do_nuke_tile`

- Milestone/step: M7.5 nuclear effects.
- Symbol name: `Unit_do_nuke_tile`.
- Reason existing hooks are insufficient: delivery animation, missile despawn,
  fallout, and victim-despawn hooks do not prove detonation; this method is the
  exact authoritative successful-strike branch and is also used by multiplayer
  replay.
- Required CSV capability: `inlead`.
- C signature: `void (__fastcall *)(Unit * this, int edx, int tile_x, int tile_y, int affected_civs)`.
- Supported executable addresses: GOG `0x5B4070`; Steam `0x5C29C0`;
  PCGames `0x5B3D80`.
- Call sites or vtable slots: function entry; called by `Unit::nuke_tile` in
  offline play and by the nuclear multiplayer sync callback.
- Fallback while missing: do not start a custom detonation; leave the nuclear
  strike wholly native.
- Verification unlocked: deterministic detonation event with exact target and
  ordering before native visual/despawn/damage work in offline and network
  replay fixtures.

### Add `Unit_get_intercepted_as_nuke`

- Milestone/step: M7.5 nuclear effects.
- Symbol name: `Unit_get_intercepted_as_nuke`.
- Reason existing hooks are insufficient: the delivery clip and missile
  despawn are shared by both outcomes; this method is the exact authoritative
  intercepted branch and is also used by multiplayer replay.
- Required CSV capability: `inlead`.
- C signature: `void (__fastcall *)(Unit * this, int edx, int tile_x, int tile_y, int intercepting_civ_id, int affected_civs)`.
- Supported executable addresses: GOG `0x5B4A00`; Steam `0x5C3350`;
  PCGames `0x5B4710`.
- Call sites or vtable slots: function entry; called by `Unit::nuke_tile` in
  offline play and by the nuclear multiplayer sync callback.
- Fallback while missing: do not start a custom interception effect; leave the
  nuclear strike wholly native.
- Verification unlocked: deterministic intercepted event with target and
  intercepting civilization, with no detonation children, in offline and
  network replay fixtures.

### M9 Natural Wonders

- Begin with existing C3X `natural_wonder_configs`, `district_tile_map`, `natural_wonder_info.natural_wonder_id`, `draw_district_for_tile`, natural-wonder animation paths, and retained-label draw hook.
- Extend existing visible-scene capture and native-body suppression if these states are not yet exported. Audit a dedicated mutation/placement boundary only if normal C3X redraw invalidation cannot expose create/load/replace transitions promptly. No new Civ III symbol/address is requested until that insufficiency is demonstrated.

### M10 Constructed Wonders

- Begin with existing C3X wonder-district state and the existing tile/district draw path. Audit a dedicated city-production/building-completion boundary only if normal Civ III redraw invalidation cannot expose construction-to-completion transitions promptly.
- Audit BIQ improvement accessors only if loaded `BIC` structures cannot provide stable Great/Small Wonder identity and map-placement eligibility.

### M11 Districts

- Begin with existing C3X district configs, `district_infos`, tile state, building-radius helpers, and `draw_district_for_tile` path; these already encode `by-count`, `by-building`, culture/era, construction, abandonment, coast alignment, and special topology selection.
- Audit a dedicated district mutation hook only if existing gameplay mutations and normal map redraws cannot invalidate a captured district instance reliably. No new Civ III symbol/address is requested until that insufficiency is demonstrated.

These names are provisional decompiler-level candidates. The user should not add them yet.

## Required Request Format

When a candidate becomes necessary, the agent adds a `required_user_action` entry here and in `project_status.json` containing:

```text
Milestone/step:
Symbol name:
Reason existing hooks are insufficient:
Required CSV capability: inlead / repl call / repl vptr / define
C signature:
Supported executable addresses: Steam / GOG / PCGames (or the project's canonical columns)
Call sites or vtable slots, when applicable:
Fallback while missing:
Verification unlocked by the entry:
```

The agent must give the same concise request to the user immediately. It must not mark the dependent step complete, silently edit the CSV, invent addresses, or repeatedly ask before the audit has produced exact information.

## Milestone Rule

Every renderer step that touches Civ III integration must declare one of:

- `required_user_action: []`, with the existing symbols it relies on; or
- One or more exact blocking requests in the format above.

Completion evidence must state which patch dependencies were used and whether any audit candidate was proven unnecessary. `TEST_INJECTED_CODE_COMPILE.bat` runs after the human adds an entry and the corresponding injected patch is implemented.
# M6.6 real-terrain semantics

M6.6 requires no new patch symbol. The existing tile capture already supplies both `m49_Get_Square_RealType` and `m50_Get_Square_BaseType`. Despite the native method names, the decompiled implementations and existing C3X terrain logic establish that `m49` supplies the underlying ground and `m50` supplies the visible square category. The renderer records therefore use `m49` for `terrain_type` material selection and `m50` for `real_terrain_type` relief/water/feature composition. Forest, jungle, marsh, and volcano feature flags follow the captured visible category. A live I11 trace exposed and corrected the earlier reversed assignment. No new address is required.

## M6.7c relief ownership

- Existing symbol: `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`.
- Confirmed capability: its `flags & 0x4010` dispatch owns Civ III mountains, hills, and volcanoes after the custom terrain bitmap is inserted.
- Implemented use: after a successful custom blit, clear `0x4010` only for the exact captured screen instance of a tile whose custom relief replacement succeeded; retain the native dispatch for per-tile and whole-frame fallback and preserve every unrelated retained-layer bit.
- `required_user_action: []`. No new patch symbol or supported-build address is required.
