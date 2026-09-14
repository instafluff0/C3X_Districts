# Civ III patch dependency ledger

The accepted retained world/view/submission implementation was renderer-only. Persistent
appearance, local dependency proofs and selected passes use the existing
`Map_Renderer_m71_Draw_Tiles` / `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`
capture and composite boundaries. `required_user_action: []`; no injected source,
ABI, executable symbol, signature or supported-build address changes. The subsequent
caller-driven native handoff is recorded under Current action below.
Shared terrain connectivity and tree color/shadow instancing use the same captured
world content and pass boundaries. They add no native hooks or ABI fields;
`required_user_action: []`. The optional terrain detail policy does not alter
picking, visibility or native replacement ownership.

CPU scene-content preparation and exact ambient work ahead remain inside the DLL.
They reuse the existing render/render-view requests and native identity epochs;
no injected source, ABI field, patch symbol or address changes are needed.
`required_user_action: []`. Worker completion has no game callback or redraw path.

This is the current boundary and outstanding-request record, not a campaign
history. Read `civ_prog_objects.csv` and the actual injected wrappers before
claiming a capability is available. The camera entry and the five visual-cadence
entries below are explicit user exceptions to the CSV editing restriction. Do not
edit other CSV entries or `ref/Civ3Conquests.h`.

## Current action

Zoom preparation adds optional DLL export `c3x_renderer_prepare_view(camera_request const*, int query_only) -> int`.
The existing compositor supplies copied prospective anchors through
`prepare_custom_renderer_zoom_views`; query/admission never authorizes pixels.
Only fresh native capture can acquire a result. Existing zoom conversion and m71/m19
boundaries are reused, with no new hook, signature/address change or CSV entry;
`required_user_action: []`. A missing export retains exact demand rendering.


Prepared nearby-view delivery adds optional DLL export
`c3x_renderer_prepare_nearby_view(camera_request const*) -> int`. Existing m71/m19
capture/composition calls it after a successful current-camera composite. The DLL
copies the snapshot; later native calls acquire a validated crop through the
existing `camera_present_view` export. No callback, camera swap, CSV entry or new
native address is needed; `required_user_action: []`. Missing capability retains
the existing exact fallback and ambient queue. Validation/staging status is in the
[retained plan](retained_renderer_plan.md).

The visual-cadence candidate uses the existing `on_timer_0x9F6500` inlead,
`Animator_update` definition and m71/m19 capture/composition. Civ III's native
Timer posts coalesced `WM_USER+1` messages to its game thread; renderer completion
never posts messages or requests drawing. For selected/working custom units, a 33 ms timer opportunity alternates
native callbacks with visual refreshes, retaining a 66 ms native deadline.
Map animation keeps the current scheduler and sampling; map-only animation
does not activate this cadence.
Intermediate draws skip the single native animation advancement service and
preserve its elapsed-time accumulator. Native camera/canvas construction remains
inside `Animator_update`; no copied animator implementation is introduced.

`required_user_action: []` for the following **five** additions to
`civ_prog_objects.csv`. The user authorized adding these verified addresses and
enabling faster cadence on September 14, 2026. All five entries are now present.
The VM's installed unmodified GOG executable matches the audited binary exactly
(SHA256 `838df6f8b3518d5f5f7ff50c7c7add715628ffabbd37221d0bcc7f080afc2746`).
Steam and PCGames.de remain `0x0` until verified. These entries are GOG-only, like the existing
camera inlead; the current patcher cannot install a zero-address inlead elsewhere.

| Symbol | Kind | GOG address | Signature/type | Capability |
| --- | --- | --- | --- | --- |
| `p_main_animation_timer` | define | `0x009F6500` | `Timer *` | Use the existing native timer owner. |
| `Timer_reset_and_activate` | define | `0x006205D0` | `void (__fastcall *) (Timer * this, int edx, void * callback_fn, void * callback_param, int duration, int resolution)` | Stop/rearm using the native mechanism; restore 66 ms on disable. |
| `Units_Image_Data_advance_animations` | inlead | `0x00405FC0` | `void (__fastcall *) (Units_Image_Data * this, int edx, float elapsed, Unit ** units, int count, void * effects)` | Suppress unit/cursor/army/effect advancement only during an intermediate visual draw. |
| `p_native_timer_inhibited` | define | `0x0072C2C4` | `int *` | Honor the original callback's native suspension guard. |
| `p_native_game_ending` | define | `0x00CC37BC` | `int *` | Honor native shutdown before calling the animator. |

The two routines each have four stack arguments (`ret 16`). The decompiler omits
the fourth advancement argument; the call site and executable bytes include the
tile-effect list. `Renderer/tools/audit_native_visual_cadence.py` checks these
bytes, call targets, timer transport and guards without running the game.
Evidence: `Renderer/native/build/native-visual-cadence/`.

Fallback: without all five symbols, compile-time gates keep the current 66 ms
callback. Runtime eligibility excludes shutdown, suspension, loading, modal,
online, unfocused, drawing, pending native unit reconciliation and special
interaction modes. A stopped native timer is never resurrected. The September 14
live trace confirms 518 intermediate visual refreshes; displayed FPS and complete
interaction acceptance remain unverified. The enabled patch table
and inlead pass the approved compile/injection smoke test. The ordinary
`INSTALL.bat` now activates the eligible faster cadence without extra settings;
installation and launching the game remain user actions. This replaces the earlier audit's incomplete conclusion that crossing
50 ms necessarily runs game work on the multimedia thread: the verified normal
transport posts back to the game thread when `callback_fn_2` is null.

Finished unit-pixel preparation itself adds no native symbols. The current-camera
publication correction removes obsolete camera swapping at m71 and rejects stale
queued cameras/projections before polling. It reuses the existing camera inlead
and requires no additional table entries. Demand-priority and typed cliff-cancellation
corrections also stay inside the DLL: `required_user_action: []`.

An earlier live test failed async scrolling. The existing
`Main_Screen_Form_move_camera` hook now treats every movement as an exact barrier:
it cancels held work and lets native camera/bounds changes remain visible to the
animator and subsequent map draw. Stationary async publication remains enabled.
This reuses the same GOG inlead, signature and addresses below; no CSV changes or
new symbols are needed. `required_user_action: []`. The approved compile/injection
smoke test passes. Installation and game launch remain user actions.

The remainder of this action records the original bridge and its dependency.


The caller-driven displayed-view bridge adds the user-authorized
`Main_Screen_Form_move_camera` **inlead** directly to `civ_prog_objects.csv`.
Signature: `void (__fastcall *) (Main_Screen_Form * this, int edx, int x, int y,
int reason, bool update_bounds)`. GOG address: **0x004DF700**. Steam and
PCGames.de addresses: **0x0**, explicitly requested by the user; this entry is
GOG-only until those addresses are established. The current patcher requires a
nonzero inlead address, so this table cannot install on those other builds yet.

The unmodified GOG executable confirms the entry's camera writes at offsets
0x4E98/0x4E9C, native bounds/wrap/clamp, and the call from the existing
`bring_tile_into_view` at 0x004DFAA1. Read-only disassembly and the approved
compile/injection smoke receipt are preserved in
`Renderer/native/build/native-handoff-20260913/`.

Reason: direct native picking, unit visibility and culling also read these camera
fields. Holding only transformed overlay anchors is insufficient. The inlead
lets native relative keyboard/edge scrolling update requested intent while the
native fields continue to describe the displayed publication. Native clamp/wrap
remains in the original function. Programmatic recenter, unit-animation camera
moves and zoom retain an exact-render barrier. The existing m71/m19 boundaries
capture, validate and commit the displayed view; m21 is called through its existing
vtable for a capture-only request traversal. No completion callback or redraw hook
is added. `required_user_action: []` for the authorized GOG table addition.

Without the inlead or any optional presentation export, async mode stays disabled
and the existing exact path remains. The user requested an `INSTALL.bat`-only workflow with no environment settings;
the existing mode now defaults on when the hook and exports are available.
`C3X_RENDERER_NATIVE_ASYNC=0` remains a diagnostic opt-out. This changes only
selection in `injected_code.c`, reusing `Main_Screen_Form_move_camera` and the
existing m71/m19 capture/publication symbols; no CSV or address changes are
needed. `required_user_action: []`. Live displayed-view validation remains pending;
enabling the default does not certify actual interaction cadence. Candidate compile/replay does not authorize
installation or launching Civ III.

The native request-identity handoff changes `C3X.h`/`injected_code.c` and adds the
optional DLL export `c3x_renderer_render_view`, using the existing versioned camera
request. API-17 layouts and legacy render calls remain compatible. Existing
`load_scenario`/renderer-unload lifetime and `Map_Renderer_m71_Draw_Tiles` /
`Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` capture/composition supply scenario,
viewer and visibility observations. Native `Tile.Body.FOWStatus` and `Visibility`
are read in the existing bounded world scan; no new executable symbol, signature
or supported-build address is needed. `required_user_action: []` for patch-table
changes. The approved injected compile passes. Staging/install/live verification
remain separate and unperformed. Exact-camera misses can still block; completion
never notifies Civ III or requests a redraw.

Retained appearance/content bindings and compact occurrence records use those same
map boundaries without expanding replacement ownership. The rejected terrain batch
experiment leaves no live batching path or new patch requirement.

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
`required_user_action: []`. The explicit-identity ordinary render entry above consumes exact queued work;
these begin/poll exports are still not bound by injected code. A general camera
handoff remains constrained by native overlays/picking, not a completion-redraw hook.
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

The unit playback correction uses those same existing `Unit_tick_anim` and normal/
reduced body hooks, with unchanged signatures and supported-build addresses.
`forward_custom_unit_body` captures whether the parent display unit equals
`Main_Screen_Form::Current_Unit` (including army representatives) and supplies it
to optional DLL export `c3x_renderer_unit_draw_playback`. Signature:
`int(unit_v1 const*, void* destination, void* background, int* bounds, unsigned flags)`.
This is a DLL capability, not a new Civ III patch symbol. The bridge falls back
to `unit_draw_expanded` / `unit_draw_background` with an older DLL. New DLLs keep
the v1 struct layout and legacy explicit-cursor behavior. Native visibility,
anchors, underlay and returned dirty bounds remain unchanged.
`required_user_action: []`; no CSV changes, new hooks, timer or redraw callback.
The matching bridge is compiled by the ordinary `INSTALL.bat` workflow.


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

## Retained circular scene surface (2026-09-13)

The existing `c3x_renderer_render_view` demand/publication boundary consumes the
new retained surface through the ordinary DLL output. Existing captured anchors,
ordered ownership flags, scenario/viewer/visibility identity and unit fallback
remain authoritative. No new Civ III patch symbol, signature, address or capture
field is needed. `required_user_action: []`. This records source/replay integration;
no staging, installation or game launch is authorized by the implementation.

Incremental GPU finishing uses the same boundary, captured anchors and ordered
output ownership. It adds no callback, redraw request, asynchronous publication
mode or executable dependency. `required_user_action: []` for this continuation.
