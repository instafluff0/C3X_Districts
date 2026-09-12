# Civ III patch dependency ledger

This is the current boundary and outstanding-request record, not a campaign
history. Read `civ_prog_objects.csv` and the actual injected wrappers before
claiming a capability is available. Agents must not edit that CSV or
`ref/Civ3Conquests.h`.

## Current action

Ordinary volcano material ownership, static crater lava and caster coverage are
renderer-only. They consume captured terrain identity and topology through
`Map_Renderer_m71_Draw_Tiles` / `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`
and preserve the existing map insertion/compositing boundary.
`required_user_action: []`; no injected source, patch-table symbol, signature or
supported-build address changes are required.

Navigation dependency/scene optimization remains renderer-only. Retained caster
proofs, region contributor selection and dependency-replayed shoreline centers
consume the existing `Map_Renderer_m71_Draw_Tiles` /
`Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` capture contract. The existing optional
DLL camera queue coalesces byte-identical requests; the injected bridge remains
synchronous. The opt-in three-zoom memory tier changes existing cache caps only.
`required_user_action: []` for these changes; the existing city-HUD request below
remains separate. Candidate replay does not stage or install this code.

The user-authorized navigation production update makes the verified wave,
backdrop, receiver-index and larger-memory defaults follow the existing injected
`enable_custom_rendering_cache` → `C3X_RENDERER_WORLD_REGIONS` setting. It adds
no injected source changes, exported ABI, lifecycle epochs or completion hook.
The same map capture/composite symbols and native-directed unit draw boundaries
remain authoritative. `required_user_action: []`; an already installed current
cache switch needs only the updated DLL on the next game start. Staging status
and rollback are recorded in `navigation_continuation.md`.

The accepted mountain body contrast and slope-aware ground projection are
shader-only. They use captured ordinary-volcano coverage to preserve volcanic
material within shared relief surfaces and the existing
`Map_Renderer_m71_Draw_Tiles` / `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`
capture and composite boundaries, with no geometry, indexing, ABI, ownership or
patch-table changes. `required_user_action: []`; no executable symbol is added.

Request-scoped retained shadow-caster descriptors reuse the existing map
submission and resource-animation boundaries. They do not change Civ III tile
capture, visibility, native ownership, renderer ABI or executable hooks.
Existing symbols remain `Map_Renderer_m71_Draw_Tiles` and
`Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`; `required_user_action: []`.
Exact per-tile height/support reuse and the standalone resident-camera witness
use these same capture boundaries without ABI or patch changes;
`required_user_action: []`. The witness does not install hooks or launch Civ III.
Opt-in guarded dirty-block/reflection clipping also stays within the existing
off-screen submission and composite boundaries, without new symbols or changed
native ownership; `required_user_action: []`.

Beach-only coastal waves are DLL-only. They consume the existing captured world
topology, tile anchors and presentation clock at `Map_Renderer_m71_Draw_Tiles` /
`Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`, and the existing
`on_timer_0x9F6500` / `QueryPerformanceCounter` ambient redraw path. The output
`visible_animation_count` and `request_continuous_redraw` fields already reach
the injected compositor. No live ownership expansion or new symbol is required:
`required_user_action: []`.

Lab reorganization, shared CPU geometry/query extraction, asset preparation,
shaders and off-screen cache/composition work require no new Civ III symbol.
Existing map and unit boundaries remain unchanged:

The full-size world-mesh sharing, version-checked viewport draw restore and
retained-image capacity changes are DLL-only. They use the existing map capture
and composite boundaries below. `required_user_action: []` for these performance
changes; the separate city-HUD request remains outstanding.

Direct natural-grid indexing and source-aware underlying relief queries are also
DLL-only. They preserve those same capture/composite symbols and ownership;
`required_user_action: []`. No cache increase or new executable address is needed
for these cold-view optimizations.

Underlying ground-grid retention is also DLL-only and uses the same existing
capture/composite boundaries. It reallocates the existing CPU cache tier while
world GPU sharing is active, validates captured dependencies and preserves raw
wrapped occurrence identity. `required_user_action: []`; no new patch capability,
signature or supported-build address is required.
Exact nested-grid sample reuse has the same DLL-only boundary and unchanged
cache ceilings: `required_user_action: []`. It needs no Civ III patch-table edit.
The flat-height shoreline certificate is also renderer-only, retaining the
existing capture and invalidation inputs: `required_user_action: []`.
Opt-in immutable completed-frame publication is internal to the existing
renderer render/blit ABI. It does not yet enable background presentation and
requires no new executable symbol or patch capability: `required_user_action: []`.
UI-owned GDI blit lifetime and the documented borrowed-output lifetime also keep
the existing ABI and native map-composite boundary: `required_user_action: []`.
The experimental DLL-only `c3x_renderer_camera_begin`, `c3x_renderer_camera_poll`
and `c3x_renderer_camera_cancel` exports add a bounded latest-request queue.
They are not Civ III symbols, require no executable addresses, and are not bound
by the injected bridge yet. Existing synchronous hooks remain unchanged;
`required_user_action: []`. Current-camera previews and native final-redraw
coordination remain implementation work, not an additional patch-table request.
Opt-in terrain-only provisional images add a distinct `PREVIEW` result to those
experimental exports, not a success result on the synchronous map API. They
claim terrain ownership only; the injected bridge does not consume them yet.
No new Civ III symbol or address is needed: `required_user_action: []`.

- Existing symbols: `Map_Renderer_m71_Draw_Tiles`,
  `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`,
  `Main_Screen_Form_tile_to_screen_coords`, `Unit_tick_anim`,
  `Sprite_draw_unit_body_normal`, `Sprite_draw_unit_body_reduced`,
  `on_timer_0x9F6500`, `QueryPerformanceCounter`, `OutputDebugStringA`.
- `audit_candidates: []` for the current renderer work.
- `required_user_action: [Main_Screen_Form_tile_to_screen_coords: define -> inlead]`
  for native city-HUD anchor alignment; details and fallback are below.

Stepped custom zoom uses the existing `Main_Screen_Form_handle_key_down` inlead
and consumes `Z` before Civ III's native two-level toggle. Existing
`Main_Screen_Form_get_tile_coords_under_mouse`, left/right-click wrappers, hover
wrapper and `Sprite_draw_on_map` inlead provide inverse input and native overlay
placement. Native unit health/status alignment uses the existing `Unit_tick_anim`
inlead. `audit_candidates: []`.

Native UI unit portraits remain outside zoom suppression. The existing
`Sprite_draw_unit_body_normal` and `Sprite_draw_unit_body_reduced` wrappers
suppress fallback only on the canvas scoped by `Unit_tick_anim`, independently
of custom unit availability. HUD/city-screen calls outside that scope retain
their original native arguments and return values. No new symbols, signatures
or addresses are required for this correction: `required_user_action: []`.

The centered five-level range (64, 96, 128, 160, 192 pixels, normal at 128)
changes only the existing key handler's step array and native-sync bounds.
Those existing symbols and capture/overlay hooks already carry numeric scales;
this range change needs no new entry, signature or address.
`required_user_action: []` for the range change itself. The separate city-HUD
row capability request below remains outstanding.

Native city names and state labels calculate their anchor through the already
recorded `Main_Screen_Form_tile_to_screen_coords` symbol. Change that existing
row from `define` to `inlead` so the implemented contextual patch can transform
city-HUD anchors. Signature:
`void (__fastcall *)(Main_Screen_Form *, int, int, int, int *, int *)`.
Recorded addresses are GOG `0x4E3B10`, Steam `0x4EC360`, and PCGames.de
`0x4E3BD0`; these are existing checkout data, not new address claims. Capability:
native city-HUD anchor transformation during custom zoom. Reason: the native HUD
calls this function before drawing city text and state icons. Fallback: leave the
row as `define`; terrain, input and unit-status zoom continue to work, but native
city HUD remains at its binary native-zoom anchors.
`required_user_action: [Main_Screen_Form_tile_to_screen_coords: define -> inlead]`.

The experimental `Main_Screen_Form_process_mouse_wheel` row is currently
`ignore`, so it cannot install the abandoned wheel patch. It may remain ignored
or be removed by the maintainer. `required_user_action: []`.

The three GOG unit inleads below are already supplied; they are not outstanding
requests. Other-build unit addresses remain unverified. The previously recorded
three effects requests remain deferred and unchanged below; this cleanup does
not authorize implementing nuclear/transient effects or ask for those edits now.
Fog-edge work and wonders/Districts remain deferred.

## Available boundaries

The approved unit fidelity update uses the existing `Unit_tick_anim`,
`Sprite_draw_unit_body_normal` and `Sprite_draw_unit_body_reduced` boundaries,
with unchanged patch capabilities, signatures and supported-build addresses.
The optional DLL export `c3x_renderer_unit_draw_expanded` returns the complete
body rectangle after a successful draw. The bridge unions that rectangle into
the parent display unit's existing `Body.Rect`, including army-member bodies;
native animation cleanup owns its next erase. Pack-authored minimum canvases
preserve the original screen anchor and permit anatomy-sized long weapons.
Legacy DLLs retain the old draw callback and rectangle behavior. Failures retain
native fallback, and no CSV symbol or patch-table edit is needed.
`required_user_action: []`. The renderer and updated injected bridge were
installed together after the user authorized the update and exited Civ III.
The separate outstanding requests elsewhere in this ledger are unchanged.

| Responsibility | Existing symbol | CSV capability / use |
| --- | --- | --- |
| Retained map lifecycle | `Map_Renderer_m71_Draw_Tiles` | `repl vptr`; bounded capture, reset and composite lifecycle |
| Tile state and insertion | `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` | `repl vptr`; authoritative anchors, state, clips and exclusive custom map plane |
| Native sprite census | `Sprite_draw_on_map` | `inlead`; not a unit-body boundary |
| Main-map stepped zoom | `Main_Screen_Form_handle_key_down` | existing `inlead`; consumes `Z` while custom zoom is enabled |
| Forest/jungle/swamp census | `Map_Renderer_m08_Draw_Tile_Forests_Jungle_Swamp` | `repl vptr` |
| Resource census | `Map_Renderer_m09_Draw_Tile_Resources` | `repl vptr` |
| Irrigation census | `Map_Renderer_m11_Draw_Tile_Irrigation` | `inlead` |
| Tile-building census | `Map_Renderer_m12_Draw_Tile_Buildings` | `repl vptr` |
| Routes | `Map_Renderer_m52_Draw_Roads`, `Map_Renderer_m52_Draw_Railroads` | `repl vptr` |
| Redraw requests | `on_timer_0x9F6500` | `inlead`; mark ordinary Animator work, never capture/render/blit here |
| Animator refresh | `Animator_update` | `define`; callable, not entry-patchable |
| One-shot actions | `Animator_play_one_shot_unit_animation` | `inlead` |
| Movement | `Unit_move`, `Unit_move_to_adjacent_tile` | `inlead` |
| Attack helper | `Unit_play_attack_animation` | `define`; callable, not entry-patchable |
| Selection | `Main_Screen_Form_set_selected_unit` | `inlead` |
| Combat | `Fighter_begin`, `Fighter_animate_start_of_combat`, `Fighter_fight` | `inlead` |
| Bombard resolution | `Fighter_do_bombard_tile` | `inlead` |
| Impact loads | `Units_Image_Data_load_animated_effect` | `inlead` |
| Removal | `Unit_despawn` | `inlead` |

`QueryPerformanceCounter` and `QueryPerformanceFrequency` are existing imports,
not CSV requests. Injected diagnostics use the existing `p_OutputDebugStringA`.

`Main_Screen_Form_tile_to_screen_coords` is an existing callable:
`void (__fastcall *)(Main_Screen_Form *, int edx, int tile_x, int tile_y, int *, int *)`,
GOG `0x4E3B10`, Steam `0x4EC360`, PCGames `0x4E3BD0`.
Logical anchors must agree with captured native anchors before extending a halo;
a mismatch rejects that extension. The worker consumes immutable snapshots and
never reads Civ III pointers. Coast/river dependencies, bounded geometry and
pixel caches, invalidation and GDI bitmap inspection remain DLL responsibilities.

## Current map-state contracts

The decompiled terrain accessors' names are misleading: `m49_Get_Square_RealType`
supplies underlying ground, while `m50_Get_Square_BaseType` supplies the visible
square category. Capture maps them to `terrain_type` and `real_terrain_type`
respectively. Relief/vegetation/water composition follows the latter.

Existing m19 capture, visibility-aware tile accessors and loaded BIC/city/leader
structures supply road/rail topology, improvement flags, resource identity,
city culture/era/size/capital/walls, river masks and active volcano state.
Canonical tile identity and wrapped screen occurrences remain distinct.
Caster/prefetch-only records must publish zero replacement flags.

Custom-on m19 ownership is exclusive: capture, load, render, validation, device,
blit and reentrant failures do not replay the multiplexed native tile renderer.
Config-off retains the original path. The older partial-relief `0x4010` masking
experiment is not authority to reintroduce fallback. Native overlays, labels,
fog, selection and HUD retain their existing ordering and ownership.

Existing infrastructure inputs also cover goody visibility
(`m15_Check_Goody_Hut`), colonies (`Tile_has_colony`, `p_colonies`,
`Tile_Building_Body`), and barbarian camps (`m7_Check_Barbarian_Camp`,
`m44_Get_Barbarian_TribeID`). Colony owner, not territory owner, selects era/tint;
camp tribe identity is not a civilization tint selector. These data sources do
not themselves approve art or expand suppression. See the corresponding current
infrastructure/source findings before extending a category.

## Goody huts and barbarian camps

The existing `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` capture/insertion hook
now captures sites through `Tile::m15_Check_Goody_Hut(viewer)` and
`Tile::m7_Check_Barbarian_Camp(viewer)`. `m44_Get_Barbarian_TribeID` supplies the
visible camp's stable composition identity. These are existing vtable calls;
no new address, signature, CSV entry or patch capability is required.

API 17 appends the camp identity and adds explicit site presence/replacement
flags. Frame, geometry and retained-background signatures observe these values.
The existing ownership validator rejects missing visible site replacements and
claims on hidden/absent sites or caster-only tiles. Config-off keeps native m19;
custom-on frame failure preserves the established exclusive-map policy.

`required_user_action`: none for symbols or patch-table edits. Deploy a matching
API 17 renderer and rebuild/install injected capture together. The approved
`TEST_INJECTED_CODE_COMPILE.bat` smoke passes with these additions. The user's
request to make both sites appear in-game authorizes this integration scope.

## Installed GOG unit hooks

Current CSV entries use these exact signatures:

```c
void (__fastcall *)(Unit * this, int edx, PCX_Image * canvas, int pixel_offset_x, int pixel_offset_y, bool include_status_layer)
int (__fastcall *)(Sprite * this, int edx, PCX_Image * background, PCX_Image * canvas, int pixel_x, int pixel_y, char * palette_path, PCX_Color_Table * color_table)
int (__fastcall *)(Sprite * this, int edx, PCX_Image * background, PCX_Image * canvas, int pixel_x, int pixel_y, int scale_x, int scale_y, int scale_divisor, char * palette_path, PCX_Color_Table * color_table)
```

The signatures correspond, in order, to:

| Symbol | Capability | GOG | Steam / PCGames |
| --- | --- | --- | --- |
| `Unit_tick_anim` | `inlead` | `0x005CBF50` | unverified; current zero entries are not patch addresses |
| `Sprite_draw_unit_body_normal` | `inlead` | `0x005F88B0` | unverified |
| `Sprite_draw_unit_body_reduced` | `inlead` | `0x005F8940` | unverified |

Reduced zoom has nine stack arguments and `RET 0x24`; the old four-argument
prototype is invalid. Native calls outside the scoped unit/canvas/current-Sprite
context pass through. Effective native palettes, background DCs and existing
dirty rectangles suffice without new hooks. The Army representative uses
`army_top_defender_id` and existing `get_unit_ptr`; its native helper needs no
additional patch. Current fallback is per intercepted body, not the earlier
proposed atomic commander/member preflight.

The source-backed boundary, audited ordinary/Army call addresses and verification
scope are in [the native unit contract](i20_native_unit_animation_handoff.md).
`tools/audit_unit_hooks.py` checks executable and CSV evidence.
Other-build audits must establish actual entry bytes and signatures before
requesting support; GOG results do not establish Steam/PCGames correctness.

## Deferred effects boundaries

Existing bombard and impact-load hooks supply begin/release/impact/outcome
observations without changing combat rolls. Impact IDs 3–6 are hit variants,
7 land miss and 8 water miss; Fire Rate may create multiple exact calls.
The effect FLC advances before Animator checks its draw-enable byte at `0x184`.
An accepted custom impact may clear only the low byte of `anim->Last` after
native load, preserving sound, waits, advancement, dirty accounting and lifetime.
No additional Animator/draw hook was found necessary.

Standalone SDI uses `Units_Image_Data_load_animation`. Nuclear delivery alone
does not prove the outcome: `Unit::do_nuke_tile` and
`Unit::get_intercepted_as_nuke` are the authoritative branches, including
multiplayer replay. Victim despawns are outcome details, not detonation triggers.
See `bombardment_and_explosion_effects.md`. Preserve these previously established
requests for that deferred scope; none blocks the current Lab workflow.

### Upgrade `Units_Image_Data_load_animation`

- Deferred scope: effects and nuclear interception.
- Symbol name: `Units_Image_Data_load_animation`.
- Reason existing hooks are insufficient: ordinary `AnimatedEffect` loads are
  already intercepted, but the audio-bearing SDI animation is a standalone FLC
  loaded directly through this function. A post-load hook is required to clear
  only its draw-enable byte after renderer ownership succeeds.
- Required CSV capability: change `define` to `inlead`.
- C signature: `void (__fastcall *)(Units_Image_Data * this, int edx, char * asset_string, FLC_Animation * anim, int civ_id, int param_4, int param_5, bool param_6)`.
- Supported executable addresses: GOG `0x4062A0`; Steam `0x406810`;
  PCGames `0x4062D0`.
- Call sites or vtable slots: function entry; existing CSV entry.
- Fallback while missing: leave standalone SDI pixels and the entire custom
  nuclear-interception presentation native.
- Verification unlocked: prove SDI pixels are absent while its native INI/FLC
  sound, animation advancement, wait, and cleanup still occur.

### Add `Unit_do_nuke_tile`

- Deferred scope: nuclear effects.
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

- Deferred scope: nuclear effects.
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


## Deferred wonders and Districts

Natural wonders start from existing `natural_wonder_configs`,
`natural_wonder_info.natural_wonder_id`, `district_tile_map` and
`draw_district_for_tile`. Constructed wonders start from wonder-district state
and loaded BIC improvement identity. Districts start from existing configs,
`district_infos`, tile/building-radius state and the current draw path.
These remain the authoritative gameplay sources; do not duplicate them.

Preserve by-count/by-building, culture/era, construction, abandonment, coast
alignment and topology contracts in `natural_wonder_rendering.md` and
`wonder_and_district_rendering.md`. Audit a mutation/placement/completion hook
only if normal state capture and redraw invalidation prove insufficient.
No new symbol is requested and this work is not authorized early.

## Rules for a new request

For each integration change, record existing symbols, concrete audit candidates
and `required_user_action` (empty when none). Do not request a speculative
function merely because it might be useful. A real request must give:

- Symbol and required capability: `inlead`, `repl call`, `repl vptr` or `define`.
- Exact C signature and supported-build addresses/entry evidence.
- Why existing boundaries are insufficient, and relevant call/vtable locations.
- Safe behavior while missing and verification unlocked by the change.

Report a new concrete request to the user in the same turn; never edit the CSV,
invent addresses or silently treat missing support as complete. Run
`TEST_INJECTED_CODE_COMPILE.bat` when the corresponding injected source changes.
DLL-only or documentation changes do not require injected compilation.
