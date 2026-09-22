# Civ III patch dependency ledger

## Retire speculative camera-image requests

`required_user_action: ["Re-run INSTALL.bat after staging the matching renderer"]`.
Existing `Map_Renderer_m71_Draw_Tiles`, `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`, and native navigation/
composition hooks keep their registered signatures and addresses. The injected
bridge no longer loads/calls optional nearby/alternate-zoom image preparation;
three function-pointer/state fields and the prospective zoom helper are removed.
Exact capture, zoom anchors, native camera following, and fallback remain in their
existing owners. The DLL retains the optional ABI symbols as explicit unsupported
responses for old bridges and recordings. No new patches or CSV changes.

## Ambient continuity through native UI and unit transitions

`required_user_action: ["Re-run INSTALL.bat to remove injected ambient pause calls"]`.
Existing `show_popup`, `Main_GUI_set_up_unit_command_buttons`, `Advisor_GUI_open`
and the existing JGL Graphsy presentation hook keep their signatures and patch
addresses. No new symbols, patch-table changes or injected state. Native popup
and command reconstruction guards still protect gameplay and optional native
redraws; they no longer pause DLL-owned ambient samples. Graphsy supplies map
visibility/loading policy. The DLL preserves the last published unit pose until
native composition replaces it and admits an eligible canvas at its first actual
stroke. Genuine escaped CPU/DC lifetimes retain their original fallback.

## Pre-configuration and pre-publication line initialization

`required_user_action: ["Re-run INSTALL.bat to update the injected bridge"]`.
Reuses the existing `OpenGLRenderer_initialize` and `OpenGLRenderer_draw_line`
inleads and style wrappers listed below. Signatures remain
`int (__fastcall *)(OpenGLRenderer*, int, PCX_Image*)` and
`void (__fastcall *)(OpenGLRenderer*, int, int, int, int, int)`. Registered
GOG/Steam/PCGames.de addresses remain `0x0062B1A0 / 0x0064E220 / 0x006274B0`
for initialization and `0x0062B450 / 0x0064E510 / 0x00627760` for drawing.
No new patch, CSV entry or injected state.

Main-screen startup (`FUN_004e2b00`) initializes lines on its full-screen canvas
before scenario configuration and the first GPU map publication. Installed GOG
bytes confirm the startup call at `0x004C8E09` and line initialization at
`0x004E2B7F`, through the already-installed hook at `0x0062B1A0`. The base
configuration still disables rendering at this point. Checking custom-on alone
therefore missed this public DC acquisition and permanently revoked later
map-copy admission, even with the previous bridge correctly installed.

The existing loaded-configuration list distinguishes base defaults alone from
loaded configuration files. Before configuration, and when custom-on,
initialization records the target without acquiring a DC. The first actual CPU
stroke initializes the original backend and replays its style. Configured-off
initialization remains immediate. Pointer/DC escape barriers remain unchanged.
The real-JGL regression fails on the previous wrapper and passes with this
change, including an actual CPU stroke before configuration. No new state is
needed. The complete native fixture now begins with this configuration sequence.

## Movement visibility invalidation

`required_user_action: ["Re-run INSTALL.bat to update the injected bridge"]`.
Reuses existing `Map_Renderer_draw_fog` (GOG `0x004C4EF0`, other builds
unregistered) and `on_timer_0x9F6500` (GOG/Steam/Complete
`0x004DE5C0 / 0x004E6FA0 / 0x004DE680`). Signatures remain
`void (__fastcall *)(Map_Renderer*, int, int, PCX_Image*, RECT*)` and
`void (__stdcall *)(void)`. No patch-table entry or injected state is added.

The suppressed native fog draw compares current native visibility with copied
rendered tile records. A difference queues one authoritative map capture. The
existing timer sets Animator's audited dirty byte before calling the original
timer, because Animator clears its dirty byte after the movement fog call.
Unchanged visibility does not request a redraw; config-off calls native fog.
The extracted reveal/conceal and timer tests pass, as does the approved injected
compile smoke test. Live movement acceptance remains pending.

## Fullscreen retained-composition memory repair

`required_user_action: []` for patch registration. This DLL-only continuation
reuses the existing native image copy/save/restore hooks, Graphsy presentation,
map prepare/commit and visual-frame exports. No signature, address, patch-table,
injected state or additional hook changes. The installed startup transfer fix is
confirmed by capture `20260920-230709-cb9049`; no reinstall is required for this
continuation after that bridge is installed.

Equal-coordinate/same-format replay copies now share immutable source patches.
The live-image ceiling is 128 MiB, including fullscreen old/new map overlap;
retained-history and replay-scratch ceilings stay at 128 MiB each. Failed replay
history is discarded while the completed display is preserved. Native CPU-access
barriers and fallback semantics remain unchanged.

## Native startup screen-transfer ownership repair

`required_user_action: []` for symbols/patch registration. Reuses
`patch_JGL_Graphsy_present`, existing hash-verified JGL Graphsy slot 41 at RVA
`0x3baa0`, with the same `int (__fastcall *)(void*, int, RECT*)` wrapper and
`int (__thiscall *)(void*, RECT*)` original signature. No CSV entry, injected
state, native drawing suppression or new hook is added. **Re-run `INSTALL.bat`**
to install the changed wrapper along with the matching renderer DLL.

Before configuration loads, original Graphsy presentation requests image slot 10
at `0x3bab6`, binds the palette, calls `BitBlt` at `0x3bb4c`, releases slot 11 at
`0x3bb58` and returns scalar zero at `0x3bb64`. The DC does not escape this body.
The existing operation scope now identifies that private transfer; the DLL
lifetime registry exempts only its caller-thread DC borrow. Pixel/bits escapes,
unscoped DC access and foreign-thread access still invalidate evidence. This
exception changes lifetime evidence only: the normal DC materialization barrier
still executes before native drawing. The wrapper restores the previous scope.

The old hook/DLL fails the new native-before-configuration startup assertion in
`native/build/live-native-startup-negative/receipt.json`. The previous fixture
exercised configured CPU snapshots only and missed this earlier native transfer.

## Native map-outline ownership repair

`required_user_action: []` for symbols/patch-table entries. Reuses the existing
`OpenGLRenderer_initialize`, `OpenGLRenderer_draw_line`, `OpenGLRenderer_set_color`,
`OpenGLRenderer_set_opacity`, `OpenGLRenderer_set_line_width`,
`OpenGLRenderer_enable_line_dashing` and `OpenGLRenderer_disable_line_dashing`
inleads with their existing signatures and addresses. No CSV edit or added
executable hook is needed. **Re-run `INSTALL.bat`** to install the changed bridge;
replacing only the renderer DLL leaves the unconditional native DC acquisition.

The native m71 tail calls `FUN_004e48d0`, which initializes the line renderer
before checking whether any colonies need outlines. `OpenGLRenderer::initialize`
requests image vtable slot 10 (public DC) before doing any drawing. The existing
PCGames.de initializer is `0x006274B0`; drawing/style entries are `0x00627760`,
`0x006277A0`, `0x00627880`, `0x006278C0`, `0x006278E0`, `0x00627900` respectively.
This destroys the map's exclusive lifetime evidence even when no lines follow.

The bridge keeps the current native line target and copies endpoint/style values
to the DLL's retained GPU overlay pass. Only already-owned targets qualify.
Public DC/pixel escapes still revoke ownership; a mid-scope escape or config-off
drains first, initializes the native backend and replays its style. GDI+ fallback
also goes through public DC access instead of bypassing the barrier via the HDC
field. Native UI and unsupported targets retain their existing backend.

## Route-cursor ABI crash correction

The live city-close/worker-selection report ended immediately after entering
go-to preview. Machine-code inspection establishes that GOG `0x004E45E0` ends
both branches with `RET 8` (`0x004E465C`, `0x004E46B3`); its caller supplies two
coordinates and does not clean them up. The renderer replacement incorrectly
used `__cdecl`, leaving eight bytes on the caller's stack even when it suppressed
the native cursor. It now uses `void __stdcall (int x, int y)` and explicitly
casts the original entry to the same ABI for configuration-off/native fallback.

`required_user_action: []` for the working fix: it handles the existing patch-table
type without editing the CSV. The table's `Main_Screen_Form_draw_route_cursor`
declaration should subsequently be corrected from `void (__cdecl *)(int, int)`
to **`void (__stdcall *)(int, int)`**; capability remains `inlead`, GOG address
`0x004E45E0`, Steam/other `0x0` (unverified). No new symbol or address is needed.
The explicit forwarding cast is safe with either table declaration. Re-run
`INSTALL.bat` to replace the installed injected wrapper; a DLL replacement alone
cannot repair the native calling convention. Actual crash-address attribution
and live-game confirmation remain separate from this proven ABI defect.

## M3.8 whole-world input

`required_user_action: []`. No new executable symbols, signatures or addresses.
The existing tile reader now also supplies complete appearance without unit-list
traversal to the optional DLL `c3x_renderer_set_world_capture` callback. The DLL
requests at most 128 records per caller-thread opportunity and copies them into
the existing scene publication; background workers receive immutable values only.
Existing map/viewer/visibility and configuration scopes reject obsolete pages.
The callback is registered at the existing renderer-load boundary and retired
on unload. Builds without it retain demand capture. Native camera-cutover work
is still unfinished; no speculative patch dependency is requested.

Cutover audit: GOG Main Screen `m22` resolves through the installed executable's
vtable to `0x004EE7A0`; it marks Animator dirty rather than drawing a fresh visual
frame. Native `Animator::update` (`0x004EEC40`) also advances actions/FLC state,
so invoking extra updates is not a safe visual-only completion boundary. The
existing exact centering guard remains required until a draw-only native overlay
refresh has been established. Shared rigid geometry and compressed backing are
DLL-only changes and add no patch-table dependencies.

## M3.7 asynchronous recovery

`required_user_action: []`. No new CSV entries, signatures or addresses.
Existing `Animator_update_display`, `Main_Screen_Form_move_camera`,
`Main_Screen_Form_center_camera`, `Map_Renderer_m71_Draw_Tiles`, and audited JGL
image hooks remain the boundaries. A shared injected dispatch helper settles an
eligible pending camera before config-off native image access; scene unload
cancels it. Viewer changes discard it. Unload/map drawing now honor failed DLL
ownership barriers. Queue/copy/admission recovery and checked display draining
remain in the DLL. Native Animator, selection centering and config-off arguments
are unchanged. An irretrievable GPU-only native surface remains fail-closed;
there is no stale CPU-pixel or custom-on native-terrain fallback.

## M3.6 native navigation boundary

`required_user_action: []`. The user explicitly authorized needed GOG patch-table
entries. Added `Animator_update_display`, an **inlead** at GOG `0x004EEC40`, with
signature `void (__fastcall *)(Animator *this, int edx)`. Steam/other build columns remain `0x0`; this adds no support claim for those
executables. Without these inleads the bridge retains its exact path. The
existing `Animator_update` callable definition remains available. The inlead
allows polling before `Animator::update` copies camera/erase/wrap fields or clears
unit canvases, then always forwards to the original function. The approved
injected compile/injection smoke test verifies the GOG entry.

Also added `Main_Screen_Form_center_camera`, an **inlead** for native
`bring_tile_into_view`, GOG `0x004DF9E0`, signature
`void (__fastcall *)(Main_Screen_Form *this, int edx, int x, int y, int reason,
bool update_bounds, bool force)`. Both other columns are `0x0`. It keeps a single
scoped injected boolean while forwarding the original arguments, preventing
selection/centering calls that use reason 1 from entering the pan fast path.
The existing `Main_Screen_Form_bring_tile_into_view` callable definition remains.
Fallback: the original exact renderer path; no native map fallback.

`Main_Screen_Form_move_camera` remains the only camera-normalization authority.
Only manual scrolling (`reason == 1`, no forced bounds update, outside native tile-centering) can defer,
and only when the native Animator would take its own early return. The original
move computes the destination and bounds; a capture-only m21/m19 traversal copies
the scene, and the bridge restores the displayed camera before returning. Native
picking, culling and overlay anchors therefore continue to use that camera.
Clamped no-op movements enqueue no capture or rendering.
Selection centering, animation centering, programmatic jumps and zoom keep their
native exact behavior and supersede a queued pan. No keyboard/input listener or
replacement gameplay camera was added.

The DLL `Navigation` owner copies the requested view and comparison inputs, uses
the M3.5 request/poll owner, coalesces repeated identical demand, and offers the
normalized native view only when ready. Native work starting while pending takes
an exact barrier at the requested destination. The native director always runs;
no action tick is omitted. Before m71 commits, a fresh complete capture validates
ordered tiles, anchors, visibility, topology, projection and epochs. Changed input
rejects prepared coverage and uses the existing exact renderer path. No native
terrain fallback, new presenter or map readback is introduced.

The existing native Animator cadence supplies retries. Completion thread messages
remain hints; this patch adds neither a new native timer nor a callback from a
worker into game code. The complete live navigation/reset/latency campaign remains
M3.7/M3.8 work; fixture timing is not a live-game measurement.

## M3.5 native nonblocking transaction boundary

DLL-only exports `c3x_renderer_native_camera_request` and
`c3x_renderer_native_camera_poll` extend the existing `CompositionOwner`.
They use the M3.4 atomic result, copied inputs and existing commit/cancel actions.
`c3x_renderer_native_camera_message` exposes the registered completion-message ID;
the worker posts a ticket hint to the requesting thread, never calls game code.
Pending grants no replacement coverage; it never inserts pixels or invokes a
synchronous fallback. Ready adoption retains its measured session-import cost.
No injected fields, native patch, signature or executable address changes are
needed for this boundary: `required_user_action: []`.

At the M3.5 checkpoint, the live injected caller used exact PREPARE until M3.6 coordinated native
Animator camera/erase/wrap state, overlays and picking. Do not replace only m19's
render call: Animator::update has already changed its camera canvases before m71,
and later unit/UI operations can otherwise join the pending worker or mix views.
The existing `Main_Screen_Form_move_camera`, m71/m19 and JGL seams remain the
starting points for that integration; this records no speculative new hook.


## M3.4 atomic GPU view publication

The optional DLL-only `c3x_renderer_gpu_camera_poll_view` export returns the
adopted image and copied view/coverage identity under one call gate. Existing GPU
and injected ABI layouts, native hooks and exact-camera synchronization are
unchanged. No new executable address is required; `required_user_action: []`.

## M3.3 demand priority and bounded world preparation

The existing DLL world compiler now retains its bounded worker pool and validated
owned results across frame leases. Current captured occurrences precede guard
content, without changing native occurrence order. Existing
`Map_Renderer_m71_Draw_Tiles`, `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`,
`Main_Screen_Form_move_camera` and JGL composition boundaries are unchanged;
the native camera still waits for its exact view. No injected source, ABI or
executable-address changes; `required_user_action: []`.

## M3.2 replaceable resident GPU camera requests

The production `c3x_renderer_gpu_render` DLL entry now joins the shared bounded
camera queue and adopts its completed resident output. Optional DLL begin/poll
exports expose replacement; no Civ III function or signature changes. Existing
`Map_Renderer_m71_Draw_Tiles`, `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`,
`Main_Screen_Form_move_camera` and native JGL composition boundaries still own
capture and exact synchronous presentation. Supersession preserves durable scene
changes and the adopted native map. No injected source or patch-table changes;
`required_user_action: []`. Native nonblocking presentation remains M3.5 work.

## M3.1 authoritative change publication

The DLL copies/version-stamps existing map captures before replaceable camera
work, then adopts coalesced changes into the existing retained world between jobs.
Existing `Map_Renderer_m71_Draw_Tiles`, `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` and
`Main_Screen_Form_move_camera` capture/camera boundaries remain authoritative;
`Unit_tick_anim` and `Sprite_draw_unit_body_normal` / `_reduced` retain unit
lifecycle capture. No native pointer enters the journal. Native suppression,
configuration-off forwarding, picking and the synchronous camera barrier are
unchanged. No ABI, injected source or executable-address change is required;
`required_user_action: []`. Candidate verification does not stage, install or
launch Civ III. Later M3 view/presentation work must preserve these contracts.

## M2.6 scheduling/reuse and M2.7 acceptance

DLL-only retained material selections, collected pose preparation, shared body
contributions and composition storage reuse consume the existing immutable inputs.
The existing `Unit_tick_anim`, `Sprite_draw_unit_body_normal`,
`Sprite_draw_unit_body_reduced` and native JGL composition hooks are unchanged.
Visibility, native action cursors, underlay ordering and configuration-off behavior
retain their current contracts. No ABI, injected code or patch-table change;
`required_user_action: []`. Candidate verification does not install or launch Civ III.

## Water scene effects and retained reflections (2.4 / 2.6)

DLL-only shoreline, open-water and directional-river execution consumes existing
API 18 immutable world topology, map visibility, native anchors and captured
presentation time. River direction is renderer-owned visual data derived from
that topology; it adds no gameplay authority or native capture. Existing map capture/composition
and visual-frame hooks remain unchanged. 2.6 retains resolved mirror samples and
rearms the DLL-owned visual timer after each callback; native gameplay timers
and injected hooks are unchanged. No native suppression, ABI change or
CSV entry is needed; `required_user_action: []`. The native waves/reflections
configuration flags retain their current meaning. Candidate verification does
not stage/install binaries or launch Civ III.

## Direct unit scene execution and composition (2.2–2.3)

2.3 removes optional map-depth provenance and keeps units above all map geometry,
with existing pose-local shadows and native composition. No ABI change is needed.
DLL-only execution extends the existing `Unit_tick_anim`,
`Sprite_draw_unit_body_normal`, `Sprite_draw_unit_body_reduced` and resident JGL
image bridge.
Copied inputs, native anchors, fog eligibility, UI scope and config-off forwarding
retain their existing ABI and ownership. No new native suppression or patch-table
entry is required. `required_user_action: []`. Candidate verification does not
stage/install binaries or launch Civ III.

## Map fog replacement (GOG, API 18)

Explicit user authorization permits this new GOG CSV entry. The direct inlead
`Map_Renderer_draw_fog` replaces `Map_Renderer::FUN_004c4ef0` at **0x4C4EF0**;
Steam and PCGames.de remain **0x0 / unverified**. Its signature is
`void (__fastcall *)(Map_Renderer *, int edx, int viewer, PCX_Image *, RECT *)`.
The entry `83 EC 44 53 8B D9` is six complete bytes; `RET 0x0C` at `0x4C555D`
confirms three stack arguments. `tools/audit_native_fog.py` verifies these bytes
and the fog-sprite call at `0x4C5510` in the local GOG executable.

Existing m19 capture supplies normalized visibility using C3X's existing
`is_explored`, `patch_Leader_is_tile_visible`, city spotlight and debug rules.
It cannot cleanly suppress the separate native fog pass. The new wrapper returns
in custom mode and otherwise calls the original function with unchanged arguments.
API 18 requires the matching DLL: final map output includes renderer fog, and
capture/presentation failure retains the existing exclusive custom-map failure
policy. Renderer-off uses native fog. No native fog is layered over custom output.
The DLL owns coverage, feathering, clipping and final GPU/CPU output. No gameplay
visibility, picking, unit HUD, labels, cursor or route ownership moves here.

`required_user_action: []` for patch-table edits; the authorized row is present.
Candidate verification does not stage/install either binary or launch the game.
A matching renderer and injected build must be deployed together when authorized.

The existing `Unit_tick_anim` inlead (GOG **0x5CBF50**) now returns before all
map drawing when the captured unit tile is not visible. The decompiled routine
owns body, civilization marker, cursor and status drawing, not native action
advancement. The existing body forwarding export also carries `UNIT_HIDDEN` so
the DLL retires cached selections and returns empty bounds. Native config-off
forwarding is unchanged; no extra unit CSV entry is needed. Tests cover hidden
whole-call suppression, scope restoration and CPU/GPU no-pixel results.
`required_user_action: []`.

## Zoom input and specific map-UI hooks

Selected-unit click repair changes only optional compatibility scheduling: no
ambient redraw during a processed mouse press. Full map redraw calls Main_GUI
draw, which rebuilds command buttons; `Base_Form::impl_m01_Show_Enabled` clears
native press owners at GOG `0x60504D` (byte-audited). The former selected-unit
exception and time-based guards are removed. Existing click/hover/selection
traces remain for live confirmation; `Main_Screen_Form_issue_command` is a
define, not a native interception point. No patch-table changes are required
(`required_user_action: []`). Resident-unit admission and scratch fixes use the
same `Unit_tick_anim` / `Sprite_draw_unit_body_normal` / reduced-body hooks and
JGL image bridge; they require no additional patches or native ownership.

Explicit user authorization covers the rows below. GOG calls, entry bytes and
stack cleanup are checked by `audit_native_visual_cadence.py`; CSV wiring and
compiled coordinate/capture behavior are checked by `test_custom_zoom.py`.
All newly added rows have Steam/PCGames.de `0x0`. Their equivalent call sites need
separate evidence before this correction is claimed on those builds.

Input uses the existing `Main_Screen_Form_get_tile_coords_under_mouse` inlead
(GOG `0x4E3C60`, existing Steam `0x4EC4B0`, PCGames.de `0x4E3D20`), signature
`int (__fastcall *)(Main_Screen_Form *, int, int, int, int *, int *)`.
Events/stored mouse coordinates remain display pixels; only picking applies the
inverse. City work-area input bypasses it. Four `repl call` rows named
`Main_Screen_Form_get_tile_coords_for_map_clip` at `0x4C32A8`, `0x4C32C4`,
`0x4C4F44`, `0x4C4F60` preserve native clip queries, including incremental draws
outside m71. C3X direct input consumers call the wrapper explicitly.

Map UI uses specific call replacements, leaving shared drawing routines callable:

| Wrapper / purpose | Original GOG function | GOG `repl call` sites |
| --- | --- | --- |
| `Main_Screen_Form_city_hud_coords` / city-label attachment | `Main_Screen_Form_tile_to_screen_coords`, `0x4E3B10` | `0x4E571E` |
| `Unit_draw_map_status` / health, status icons, stack marks | `Unit_draw_status` (`FUN_005ba750`), `0x5BA750` | `0x5CC41D`, `0x5CC9EB` |
| `Animator_draw_map_unit_cursor` / selected-unit cursor | `Animator_draw_unit_cursor` (`FUN_004f03e0`), `0x4F03E0` | `0x5CC2B1`, `0x5CC7F8` |
| `Sprite_draw_map_unit_marker` / civilization marker | `Sprite_draw_scaled_color` (`FUN_005f84b0`), `0x5F84B0` | `0x5CC29D`, `0x5CC7D8` |

The paired unit sites cover ordinary and army drawing. Signatures (including
fastcall's unused EDX slot) are:

- City coordinates: `void (Main_Screen_Form *, int, int, int, int *, int *)`.
- Status: `void (Unit *, int, PCX_Image *, int x, int y, bool stack_marks)`.
- Cursor: `void (Animator *, int, int x, int y)`.
- Marker: `int (Sprite *, int, PCX_Image *, int x, int y, int color, int scale_x, int scale_y, int divisor, PCX_Color_Table *)`.

The marker has **eight stack arguments**, confirmed by argument loads and
`RET 0x20`; its decompiler signature is incomplete. Status returns with
`RET 0x10`, cursor with `RET 0x08`. Wrappers preserve all non-position arguments.
Each attachment point is transformed once, preserving native pixel-sized UI
offsets. City HUD's internal coordinate call replaces the former global
`tile_to_screen_coords` inlead and context flag; that symbol is now `define`
again, with its existing other-build addresses unchanged. `Unit_tick_anim`
retains capture/canvas scope only. Its offset translation, undo path and three
zoom fields are removed. City-screen status callers remain unpatched.

`required_user_action: []`. Config-off forwards unchanged arguments. No new timer,
redraw request, installer execution or gameplay launch is part of this change.

## Renderer-owned visual scheduling

Existing `on_timer_0x9F6500` retains original gameplay advancement. The existing
`Units_Image_Data_advance_animations` trampoline now always forwards; the former
33/66 ms native timer split and visual-only `Animator_update` call are retired.
No CSV signature or address changes are needed. Existing unit/map captures bind
the optional DLL export `c3x_renderer_visual_clock` (`c3x_renderer_i64 (*)(void)`),
sharing the renderer's QPC-frequency visual timeline. Older DLLs retain their
existing capture clock.

The existing popup, Advisor and command-button function scopes supply explicit
entry/exit pause policy, including exits with no further screen transfer. The
hash-verified Graphsy final-transfer hook supplies visibility policy and seals
completed rectangles. The renderer uses
Win32 `SetTimer`/`KillTimer` on the existing presenter thread; it never requests a
Civ III redraw. Original native demand remains the compatibility path while no
retained GPU front is ready. Existing `Unit_despawn`, capture, graphics-load/unload
and CPU-access boundaries preserve retirement and config-off ownership.
`required_user_action: []`; normal authorized DLL staging does not install or
launch the game. See [visual frame ownership](visual_frame_ownership.md).

## Unit shadow pass

The resident unit owner now submits immutable caster records through its existing
GPU worker. Existing native unit capture/draw and final composition hooks are
unchanged; no executable/JGL signature or address is added. CPU destination
fallback remains with its existing owner. `required_user_action: []`.

## Native graphics lifetime boundary

Under the user's existing permission for necessary hook entries, two direct
GOG-only `inlead` hooks now bracket native graphics lifetime. Steam/PCGames remain
`0x0`; unsupported builds keep the existing CPU composition fallback.

| Symbol | Signature | GOG | Responsibility |
| --- | --- | --- | --- |
| `load_jgl_lib` | `void * (__cdecl *)(char const * filename)` | `0x6343D0` | Call original once, then attach tracking before window/canvas creation. |
| `unload_jgl_lib` | `void (__cdecl *)(void)` | `0x634390` | Drain before native destruction, restore hooks, retire tracking, preserve original unload. |

`audit_native_composition.py` verifies both complete six-byte prologues, the
factory-result store/return, original unload and exact table entries against the
preserved GOG executable. Failed loading attaches nothing; failed GPU drainage
preserves the existing unload guard. A DLL reload starts with empty lifetime
evidence. `required_user_action`: normal `INSTALL.bat` and fresh-process game check;
no additional address action. There is no polling, timer or redraw notification.

The existing hash-verified JGL image table additionally wraps slot 59, RVA `0x1ca0`,
`void (__fastcall *)(JGL_Image *, int edx, void * palette)`. Its original palette
binding remains authoritative; private DC access is attributed to metadata, while
public DC/pixel escapes still revoke eligibility. Same-size native image init is
a no-op and cannot establish a new storage lifetime after an escape.

The accepted UI fix remains: borrow Graphsy at DLL RVA `0x70d30`; never call the
misnamed `get_graphsy_object_ptr` factory during hook attachment. The confirmed
factory/reset defect and exact source evidence are preserved in
`native/build/gpu-composition/ui-owner-fix-checkpoint/`. Temporary sprite diagnostic
callbacks/readbacks are removed from game execution.

## Native composition observation — September 14, 2026

The user authorized necessary verified CSV additions. Two GOG-only entries now
identify the native screen-transfer boundary; Steam/PCGames remain `0x0`:

| Symbol | Capability / signature | GOG |
| --- | --- | --- |
| `JGL_present_screen` | `inlead`; `void (__cdecl *)(RECT * rect)` | `0x606780` |
| `p_jgl_screen_canvas` | `define`; `PCX_Image *` | `0xCAD030` |

The first instruction is a complete five-byte absolute load. The original wrapper
finishes tooltip/cursor drawing to this PCX image before calling Graphsy slot 41;
`audit_native_composition.py` checks those bytes and the cdecl return against the
preserved original GOG binary. Existing m71/m19 capture identifies the map canvas
but cannot identify its later copies into the screen canvas. Both capabilities
must exist before observation attaches. `required_user_action: []`.

The permanent injected wrappers optionally observe hash-pinned JGL image slots
0/1/3/4/10/13/16/17/33/42/43/44/45/46 and sprite slot 17. They preserve native return values, storage,
leases and drawing. Pixel aliases 5/7 and bits alias 8 already call the observed
core slots; do not hook them twice. The [probe audit](gpu_composition_probe.md)
records DLL RVAs, signatures, binary hash and bounded capture. Runtime module
slots are checked before patching; the startup tracker retains them across scene unload, while observation-only attachment restores them on detach;
these DLL RVAs are not CSV addresses. No renderer callback requests a redraw.
Missing exports, unsupported executable capabilities or a mismatched DLL leave
the current CPU composition path. GPU destination substitution is enabled only after resident-map admission in the candidate source. The user subsequently authorized
staging observation DLL `038faec0…0be2e2`; the subsequent compatibility evaluation
`a79fb579…dec552` remains a preserved control; the retained plan identifies the current staged evaluation.
The independent packed GPU executor uses no additional hooks or CSV entries:
`required_user_action: []`. The production owner now consumes observed lifetime evidence; actual game coverage remains pending.
The adapter adds `patch_JGL_Image_clip`: concrete JGL slot 13, DLL RVA `0x1a40`,
`int (__fastcall *)(JGL_Image *, int edx, RECT *)`. Its audited private HDC lease
updates clip metadata only; the native body/return value remain authoritative.
This runtime slot is verified/restored with the others, not a CSV entry. The caller-thread seam also supports the live completed-screen compatibility
callback described below. Resident map demand activates exclusive destination admission.
No additional executable address is required.
Resident-map publication and composition add optional `c3x_renderer_gpu_render`
and `c3x_renderer_gpu_images` DLL exports, driven by existing `RendererWorker`.
Resident-map publication required no new executable hooks. The native image
adapter sends its tested native operations through those worker exports. Optional
image packets specify BGRA/555/565 storage and bounded ordered commands; no native
pointers leave the caller. Existing image/sprite hooks are reused unchanged.
Image slots 16/17/33 now also carry positive in-bounds stretching, palette/null
fills and keyed full-color transfers. The internal image-command packet is 96
bytes with explicit source extent; exact-size validation rejects older packets.
No native signature/address or injected hook changes; `required_user_action: []`.
`patch_JGL_Graphsy_present` now wraps concrete Graphsy slot 41 at DLL RVA
`0x3baa0`, verified against table RVA `0x685f8`. Original signature is
`int (__thiscall *)(void *graph, RECT *rect)`; injected wrapper is
`int (__fastcall *)(void *graph, int edx, RECT *rect)`. The body returns zero
and transfers screen PCX member `+0x148` through window DC `+0x138`. It runs after
the existing GOG `JGL_present_screen` wrapper finishes tooltip/cursor drawing.
The optional callback consumes the complete transfer or leaves original JGL/DC
fallback; table protection and restoration include this slot. The same hook lazily resolves the existing observation/presentation exports from
the process-owned module when configured UI demands a frame before map loading or
after scene unload. It preserves native config-off and caller-thread ownership;
no map assets, new timer or new hook are required. No CSV address is
added: it is a hash-pinned runtime DLL slot, admitted through existing GOG-only
capabilities. `required_user_action: []`. The loader now resolves
`c3x_renderer_native_image`, which routes resident images and completed CPU UI
sources through one presenter, before or after map admission. Foreground transfer
preserves queued camera input and never selects GDI solely because preparation is busy.
Observation expiry does not detach the live transfer owner; config-off/unload
releases presentation before original GDI resumes. The older staged observer is
preserved as the rollback control. The wrapper preserves original palette
binding through verified image slot 59 (`0x1ca0`) and palette-owner global RVA
`0x70f48` (native palette member `+4`). Its scoped 16-bit metadata operation may
use the native private DC without treating palette binding as a pixel escape;
other DC access still restores CPU ownership. Injected compilation passes.

The completed CPU-screen snapshot now reads JGL's logical `Bits_Data` member
(`+0x4c0`, stride `+0x40`) privately on the caller thread after `GdiFlush`.
The audited core getter at DLL RVA `0x1b70` returns that exact member while
incrementing `Bits_Data_Links`; calling its public hook for our own snapshot
incorrectly marked the eventual game screen as permanently escaped at startup.
No pointer survives the copy. The production composition owner explicitly
materializes GPU contents before CPU snapshot fallback, including failure to
allocate the optional full-color image; failed barriers deny access. Real game
bits/DC requests retain their existing ownership barriers. This DLL-only repair
reuses `patch_JGL_Graphsy_present`; no signature, hook, CSV entry or injected
state changes are needed. `required_user_action: []`.

Fullscreen 2240×1260 admission and its 2248×1268 lighting border are DLL bounds,
not new native capabilities. The existing frame dimensions, image hooks and
Graphsy transfer supply exact sizes; no injected changes or patch entries are
needed. `required_user_action: []`.

The existing `patch_JGL_Sprite_draw` now forwards typed sprite/palette/anchor
arguments to the image adapter before native fallback. JGL sprite slot 17 remains
RVA `0x8180`, signature `int (__thiscall *)(JGLSprite *, JGL_Image *, int x,
int y, void *palette)`. Ordinary 8/16-bit and row-trimmed 8-bit sources pass exact
native tests, including positive indexed scaling and destination-palette keys.
Mirrored sources and unsafe trimmed edges retain native ownership. No new table
entry or address is needed. The production owner routes these operations;
`required_user_action: []`.

HUD alpha composition uses three additional hash-verified JGL sprite slots:

| Injected wrapper | Slot / DLL RVA | Original `__thiscall` arguments after `JGLSprite *` |
| --- | --- | --- |
| `patch_JGL_Sprite_blend` | 20 / `0x9aa0` | `JGLSprite *alpha, JGL_Image *background, JGL_Image *destination, int x, int y, void *palette` |
| `patch_JGL_Sprite_blend_onto` | 21 / `0x9a30` | `JGLSprite *alpha, JGL_Image *destination, int x, int y, void *palette` |
| `patch_JGL_Sprite_alpha_onto` | 22 / `0x9b20` | same as slot 21 |

All return `int`; injected wrappers add unused fastcall `edx`. Existing GOG
`Sprite_draw_for_hud` (`0x5F83E0`) calls slot 20 for normal HUD chrome; native
`FUN_005f8450` uses slot 22 for main-screen buttons. Slots 20/21 retain their
premultiplied source/background rule; slot 22 retains straight alpha and its
native rounding. Unsupported source layout/extent restores CPU ownership.
Native fallback remains scoped private drawing. Slot 20's helper `0x10ab0`
acquires background/destination separately (`0x10bce`, `0x10beb`) but releases
both links against background (`0x10d70`–`0x10d77`). The wrapper restores the
entry `Bits_Data_Links`/`Current_Bits_Data` of both images after that scalar-only
call, preserving caller leases and avoiding false outstanding destination borrows.
Slot 21's helper `0x10d90` borrows once (`0x10eae`) but releases two links
(`0x1100c`); slot 22 borrows at `0x9bd9` and never releases. Their wrappers likewise
preserve destination entry state. Actual startup tests cover separate/aliased
images, clipped no-ops, zero outstanding borrows and pre-existing caller leases.
The table span covers every hooked slot and detach restores all three. Existing
GOG-only executable guards and pinned JGL hash apply; other builds are unverified.
No executable symbol/signature/address or CSV changes; `required_user_action: []`.

Map-label backgrounds and native borders use two additional image slots on the
same verified JGL image table (RVA `0x68238`):

| Injected wrapper | Slot / DLL RVA | Original `__thiscall` signature |
| --- | --- | --- |
| `patch_JGL_Image_tint` | 18 / `0x2320` | `int(JGL_Image *, RECT *, int color, int percent)` |
| `patch_JGL_Image_line` | 25 / `0x2160` | `int(JGL_Image *, int x0, int y0, int x1, int y1, int color, int unused)` |

The first reaches native helper `0x40c0`; native map-label code calls it at
`PCX_Image` destinations with 25/50 percent background retention. The second
reaches `0x2810` (16-bit), ignores its final argument, and underlies
`PCX_Image::draw_horizontal_line`, `draw_vertical_line`, and `draw_rectangle`.
The GPU owner preserves clipping and native palette/packed-word colors; tint
retains the independent full-color map contribution. CPU fallback remains private
native drawing. The diagonal line helper borrows twice (`0x28f3`, then e.g.
`0x29c5`) but some exits release once; its scalar wrapper restores entry bits/lease
state, as checked with caller-held pointers. Hook attachment verifies both RVAs,
and detach restores both slots.
These runtime DLL hooks do not require executable patch entries. Existing GOG-only
capability and pinned-DLL checks apply; other builds remain unverified.
`required_user_action: []`; `civ_prog_objects.csv` is unchanged.

Lookup effects add `patch_JGL_Image_lookup`: image slot 21 / RVA `0x22c0`,
`int __thiscall(JGL_Image *, RECT *, JGL_Image *background, int percent, void *table)`;
and `patch_JGL_Sprite_lookup`: sprite slot 33 / RVA `0x8fc0`,
`int __thiscall(JGLSprite *, JGL_Image *, int x, int y, void *table, void *palette)`.
The former reaches `0x4ec0` from `FUN_00600090`, including unit-control canvas
painting; it clips to image bounds (not the clip rectangle), substitutes background
for word `0x7c1f`, and indexes the caller's table. Its scalar fallback preserves
both entry leases because the native routine releases neither. Sprite helper
`0x14db0` uses indices 1–15 as lookup blocks, 0 as black, and 16–255 as no-op;
its 16-bit helper `0xaa40` and trimmed sources do not draw. Positive scaled sprites
share the audited sampling program. Mirrored lookup sprites retain native fallback.
The related native FLC form is `patch_JGL_Sprite_lookup_over`: sprite slot 35 /
RVA `0x90a0`, `int __thiscall(JGLSprite *, JGL_Image *background,
JGL_Image *destination, int x, int y, void *table, void *palette)`. Helper `0x159e0`
is reached by `Sprite::FUN_005f88b0`, including FLC map/unit-control drawing. It
uses palette indices 0–223, lookup blocks 15–30 for indices 224–239 and blocks
0–14 for 240–254; only 255 skips. A magenta destination selects the background.
It ignores mirror signs and does not draw unequal scales. Its private destination
borrows are not released; the wrapper preserves both images' entry leases.
Native 555/raw indexed input is admitted; unsupported layouts keep fallback.
`FUN_0062c090` constructs up to 31 lookup blocks from the explosion palette;
this is a generic table at runtime, not a Civ VI asset dependency.
The scaled FLC form is `patch_JGL_Sprite_lookup_scaled`: sprite slot 34 /
RVA `0x90e0`, `int __thiscall(JGLSprite *, JGL_Image *background,
JGL_Image *destination, int x, int y, int scale_x, int scale_y, int denominator,
void *table, void *palette)`. `Sprite::FUN_005f8940` reaches it for zoomed-out
unit cursors and FLC map effects (`FUN_004e45e0` and native animator callers).
Helper `0x15c90` clips first, restarts source traversal at byte zero, consumes
every second byte and adjusts source rows using the explicit scale and clipped
width. The default half-size form uses every second row. Its palette/shadow
composition shares slot 35's GPU program. Unequal axes return 1; negative explicit
scale is a native no-op. Source-bounded raw 8-bit/555 forms are admitted; unsafe
source traversals retain fallback. Both entry image leases survive scalar fallback.
All four RVAs are checked at attach and restored at detach; existing supported-build
limits apply. No executable patch entry is needed; `required_user_action: []`.

Single-key artwork, solid masks and native shadows use three verified JGL hooks:
`patch_JGL_Sprite_keyed`, slot 23 / RVA `0x8050`,
`int __thiscall(JGLSprite *, JGL_Image *, int x, int y, void *palette)`;
`patch_JGL_Sprite_mask`, slot 29 / RVA `0x8600`, the same signature with
`int color` before `palette`; and `patch_JGL_Sprite_shadow`, slot 31 / RVA
`0x8ee0`, the same signature with `void *table` before `palette`.
`FUN_005f8270` uses slot 23 for Advisor/UI artwork (only index 255 skips).
`FUN_005f8570` and its scaled wrapper use slot 29 for solid selection/picking
masks. Raw masks skip 254/255 and accept packed colors; trimmed masks skip only
255 and always resolve a palette index. `Sprite::draw_shadow_on_map` and
`FUN_005f86f0` reach slot 31 for map/resource/native-animation shadows. The native
`FUN_00403d10` allocation is four lookup blocks; admitted source codes 248–251
select these blocks and 254/255 skip. Out-of-allocation codes retain native
fallback. Trimmed shadow sources do nothing; trimmed slot 23 does nothing and
returns Y. CPU-owned picking masks stay native; admitted map composition uses
the same GPU sprite/lookup programs. Both lookup assets coexist within the existing
budget. All three slots are checked/restored at attach/detach; existing pinned-DLL
and supported-build limits apply. `required_user_action: []`; no CSV changes.


`patch_JGL_Sprite_opacity` wraps sprite slot 37 / pinned JGL RVA `0x9220`:
`int __thiscall(JGLSprite *, JGL_Image *, int x, int y, float opacity,
void *palette, int flags)`. The EXE wrapper `FUN_005f8a70` uses it for command-panel
icons and UI fades. The existing sprite source owner and GPU blend submission
preserve native sixteenth-step weights, low-byte flags, clipping and positive
scaling. The pinned binary accepts [0,1] but compares its opaque endpoint against
100 minus 1/32; opacity 1 therefore still selects 15/16. Only 255 skips normally;
a nonzero flag low byte also skips 248–254. Trimmed sources are native no-ops.
Existing CPU fallback remains for unsupported inputs. The runtime slot is
verified/restored with the others; no CSV entry or executable address is added.
`required_user_action: []`; existing GOG/pinned-DLL limits apply.

`forward_custom_unit_body` now offers the captured unit and native destination /
underlay identities through `C3X_NATIVE_UNIT_DRAW` before acquiring either DC.
The existing normal/reduced unit-body hooks and signatures are unchanged; GPU
success returns the same expanded erase bounds. The optional
`c3x_renderer_gpu_unit` DLL export reuses playback/pose ownership and composes on
the existing GPU worker. The production native owner consumes this operation for
admitted images; before admission, the CPU compatibility path remains available. No CSV change is
needed; `required_user_action: []`. Injected compilation passes.

The existing `patch_init_floating_point` now starts read-only lifetime observation
before native canvas construction. It pins one renderer DLL reference independently
of scenario/configuration ownership, and tracks at most 1024 live image identities.
Missing exports leave the older staged DLL unchanged. GPU admission uses successful
INIT evidence; external bits/DC access or a foreign thread revokes it until reinit.
Private native operations preserve it. Config-off draws still execute native code.
The connected fixture admits pre-map canvases on demand, without startup GPU storage.

These hash-verified image slots preserve native font state and offer text to composition:

| Injected wrapper | Slot / DLL RVA | Original `__thiscall` arguments after `JGL_Image *` |
| --- | --- | --- |
| `patch_JGL_Image_font` | 42 / `0x1d10` | `void *font` |
| `patch_JGL_Image_default_font` | 43 / `0x1d40` | none |
| `patch_JGL_Image_text_index` | 44 / `0x1db0` | `int color` |
| `patch_JGL_Image_text_rgb` | 45 / `0x1d70` | `int r, int g, int b` |
| `patch_JGL_Image_text` | 46 / `0x1de0` | `int x, int y, char const *text, int count` |

All return `int`; injected wrappers add the unused fastcall `edx` argument.
Slot 46 offers typed text to composition before its original private DC lease;
ordinary bounded text runs now use resident native-font response data and GPU
composition. Unsupported state/extent retains the explicit CPU barrier. Font,
layout and clipping remain native; slight text-edge color differences are authorized.
These use the audited GOG JGL hash and existing
GOG executable guards. No CSV additions; `required_user_action: []`.

`composite_custom_renderer_frame` now calls the optional DLL export
`c3x_renderer_native_map_view(int action, void *image, camera_request const *, camera_view *)`
through a cdecl pointer. PREPARE returns resident metadata and its actual sample clock; existing ownership
validation precedes COMMIT, and rejection CANCELs without inserting pixels.
No new executable address is needed. Reset/configuration/detach drain the same
native owner. Negative barrier results deny native bits/DC leases and defer native
reinit/destruction rather than exposing stale storage; device-loss reconstruction is user-deferred.
The clock-aware owner passes the injected smoke, native dispatch contracts and
connected tests at both display sizes. Older DLLs lack the new export and retain
the existing CPU fallback. `required_user_action: []`.

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
claiming a capability is available. The camera, five visual-cadence and Advisor
entries below are explicit user exceptions to the CSV editing restriction.
Do not edit other CSV entries or `ref/Civ3Conquests.h`.

## Current action

The user authorized a direct Advisor lifecycle patch for the reported flicker.
`Advisor_GUI_open` changes from `ignore` to `inlead`, wrapping the original with
save/set/restore of `custom_renderer_modal`. This suspends extra renderer scheduling
and 33 ms visual refreshes during construction and the native dialog loop, including
nested page switches. Native painting, messages and ordinary timer behavior remain
owned by Civ III. The page-specific m95 hooks cannot cover the complete lifetime.

- Signature: `void (__fastcall *)(Advisor_GUI * this, int edx, AdvisorKind kind)`.
- Existing table addresses: GOG `0x49D070`, Steam `0x4A3AF0`, PCGames `0x49D100`.
  GOG entry, stack argument, page-construction call, modal-dialog call and return
  were checked against the original executable; other mappings are unchanged and
  were not newly verified. No new address was inferred.
- Verification: nested lifecycle/cadence contract and approved injected compilation.
  Actual Advisor painting remains a live integration checkpoint.
- Missing-hook fallback: existing native rendering; renderer-added scheduling would
  lack this Advisor suspension. `required_user_action: []` under this authorization.

The same suspension now honors existing `paused_for_popup`, owned by
`patch_show_popup` / `WITH_PAUSE_FOR_POPUP`, instead of setting/clearing a shorter
pause inside `PopupForm_impl_begin_showing_popup`. That helper returns before the
outer native popup finishes layout/dialog handling. Existing
`Main_GUI_set_up_unit_command_buttons` (`inlead`, GOG `0x550BB0`, Steam `0x55BCE0`,
PCGames `0x550B60`; `void (__fastcall *)(Main_GUI * this)`) saves/sets/restores the
same renderer guard around its complete native/mod button reconstruction.
`show_popup` already has `inlead` at GOG `0x611530`, Steam `0x62DAF0`, PCGames
`0x611460`, signature `int (__fastcall *)(void * this, int edx, int param_1, int param_2)`.
These reuse existing symbols and state; no CSV changes, new timer, delayed repaint,
parent-form inspection or drawing interception. `required_user_action: []`.

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
changes; city-HUD hook activation is recorded at the top of this ledger.

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
- The approved zoom activation is listed in the shared-projection correction at
  the top of this ledger.

Stepped custom zoom uses the existing `Main_Screen_Form_handle_key_down` inlead
and consumes `Z` before Civ III's native two-level toggle. Existing
projection hooks described above and `Sprite_draw_on_map` provide inverse input
and native overlay placement. Native unit health/status alignment uses the existing `Unit_tick_anim`
inlead. `audit_candidates: []`.

The September 15 retained-unit step removes the separate custom-unit flag.
Custom-on unit map bodies cannot replay native sprite bodies, at any zoom;
renderer-off and UI portraits preserve their native routes. `Unit_despawn`
(existing inlead, unchanged signature/addresses) now calls the renderer-only
`c3x_renderer_unit_forget(int unit_id)` export before native destruction. This
retires instance/playback identity and queued prediction; it does not draw or
advance gameplay. Existing `Unit_tick_anim` and body hooks remain the authoritative
capture/order adapter pending complete scene collection. No new Civ III symbol,
CSV edit or native clock change: `required_user_action: []`.

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
`required_user_action: []` for the range change itself. Shared-projection activation is recorded above.

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

## Continuous native presentation and keyed tactical correction

Existing hooks `Animator_draw_map_unit_cursor` and
`Main_Screen_Form_update_in_go_to_mode` now pass the existing
`Base_Data.Canvas.JGL.Image` as the tactical background. The native selected
cursor (`0x4F03E0`) and route cursor (`0x4E45E0`) already use that map canvas
with the separate keyed unit canvas. Signatures, supported-build addresses and
patch entries are unchanged; no CSV edits or additional symbols are required.
Sampling, blending and lifecycle work remain in the DLL. Config-off still calls
the original functions. `required_user_action`: rerun `INSTALL.bat` after staging
the corrected candidate to install these two background arguments.

## Milestone 2.5 — tactical draws (GOG)

The user's explicit permission to add GOG functions applies to these concrete
entries. Verified against the installed GOG executable and the reference source;
other-build addresses remain zero/unverified. No executable address is inferred
from another build. `TEST_INJECTED_CODE_COMPILE.bat` passes.

| Symbol | Capability / signature | GOG address | Steam / other | Reason and fallback |
| --- | --- | --- | --- | --- |
| `Main_Screen_Form_update_in_go_to_mode` | inlead; `void (__fastcall *)(Main_Screen_Form*, int)` | `0x4E46C0` | `0 / 0` | Lexically capture existing native route line/text output; original pathfinding, turn arithmetic and action side effects still execute. Config-off or no admitted tactical owner calls original unchanged. |
| `Main_Screen_Form_draw_route_cursor` | inlead; `void (__stdcall *)(int, int)` | `0x4E45E0` | `0 / 0` | Replace destination FLC with copied anchor semantics inside route scope. Callee pops both arguments; see the ABI correction above for the existing CSV type. Config-off/no admitted owner calls original. |
| `Map_Renderer_draw_grid` | repl vptr; `void (__fastcall *)(Map_Renderer*, int, PCX_Image*, int, int, int, int)` | slot `0x66A594`, target `0x4C5570` | `0 / 0` | Suppress native grid in custom mode; copied `MapGrid_Flag` and captured anchors drive the DLL pass. Config-off calls original. No input hook. |

Existing dependencies: `Animator_draw_map_unit_cursor` call sites `0x5CC2B1` /
`0x5CC7F8`, callable `Animator_draw_unit_cursor` `0x4F03E0`, native JGL line/text
hooks, native map prepare/commit and image-lifetime tracking. Cursor eligibility
is the existing GOG `Animator + 0x1914` bit 0; no selection rule is reconstructed.
Required user action for patch registration: **none**; these GOG entries are in
`civ_prog_objects.csv` under the earlier explicit authorization. Visual acceptance,
staging and the strategic live-game check are separate from this compile proof.

### Private CPU unit fallback after transient GPU rejection

Existing symbols: `Unit_tick_anim`, `Sprite_draw_unit_body_normal`,
`Sprite_draw_unit_body_reduced`; existing helper `forward_custom_unit_body`.
The synchronous renderer fallback now saves/restores the existing
`custom_renderer_native_operation` around its paired DC leases, using the
already-audited private sprite scope. A failed GPU barrier still denies access;
real public or foreign-thread escapes still revoke lifetime evidence. No new
state, signature, supported-build address or CSV entry is required.

`required_user_action`: none for installation; the user explicitly authorized
an agent game attempt, and INSTALL displayed success for candidate `97593ec9…`.
The game reached its main menu; the user subsequently took over gameplay testing.
Approved injected compilation passes. Extracted bridge tests cover success,
renderer rejection, denied destination/background DC, zoom, portraits and scope
restoration.

### Independent ambient delivery during native UI stalls

Existing dependencies: `patch_JGL_present_screen`, `patch_JGL_Graphsy_present`,
`patch_on_timer_0x9F6500`, popup/Advisor/command-button visual policy scopes, and
`C3X_NATIVE_IMAGE_PRESENT` / `C3X_NATIVE_VISUAL_POLICY` in the renderer DLL.
No injected source, native signature, executable address or CSV entry changes.
The DLL replaces its UI-thread visual timer and HWND swap chain with an owned
cadence thread and a DirectComposition visual on the same HWND. Native actions,
scene capture, camera adoption and GDI ownership barriers keep their existing
caller-thread contracts. Presentation failure retires visual readiness so the
existing native compatibility path can recover.

`required_user_action: []`: DLL-only staging under the existing authorization;
the installed bridge is unchanged. This presentation path requires Windows 8+
and is verified on the Windows 11 VM. Automated blocked-window-thread display
proof is distinct from live-game acceptance.


### First-use native HUD destination admission

Existing dependencies: `patch_JGL_Sprite_blend`, the installed native image
lifetime/lease hooks, and `C3X_NATIVE_SPRITE_BLEND` in the renderer DLL. The
adapter extends ownership from an owned background into an eligible destination
before blending. Existing lifetime exclusions, CPU barriers and original native
fallback remain authoritative. Fullscreen live-image capacity is DLL-owned.
No injected source, signature, supported-build address or CSV entry changes.

`required_user_action: []`: DLL-only evaluation staging under existing
authorization; the installed bridge remains unchanged.


### Retained-history capacity and replay working surfaces

Existing dependencies: the installed native image operation/ownership hooks,
`C3X_NATIVE_IMAGE_PRESENT`, and renderer-owned visual delivery. The DLL raises
retained-history capacity to 256 MiB and reuses private replay textures within
the existing scratch cap. It composes fully covered base underlays once while
preserving sparse/zero coverage and aliased native operation order. No injected
source, signature, supported-build address, ownership scope or CSV entry changes.

`required_user_action: []`: DLL-only evaluation staging under existing
authorization. Native bit/DC lifetime exclusions and config-off remain intact.

### Bounded native composition recording

Existing dependencies: `c3x_renderer_native_image`,
`c3x_renderer_native_lifetime`, the GPU composition session and existing native
presenter. The opt-in recorder lives entirely under `Renderer/`, observing the
installed bridge and recording copied GPU inputs/results with local identities.
No injected code, hook signature, supported-build address, ownership scope or
CSV entry changes. Ordinary launches keep recording disabled.

`required_user_action: ["Run Renderer/CAPTURE_GAME.bat once for the requested live recording"]`.
DLL-only evaluation staging; no new INSTALL is required. Capture adds diagnostic
readbacks and is not a performance-baseline run. The helper saves the bounded
recording alongside logs automatically. See the validation document for replay
scope and the remaining native-ownership/scene/retained-animation coverage gaps.
The first live attempt (`20260921-053809-a0e677`) lost recording settings across
XP-compatibility elevation. The corrected launcher establishes them after host
elevation; its real-launcher/staged-DLL stand-in and missing-file control pass.
One replacement recording is needed. This correction changes only Renderer
tooling and documentation; the staged DLL and installed bridge remain unchanged.


### Packed CPU-source validation

Existing dependencies: the adapter's already-audited private JGL image bit
getter/release pair and existing GPU upload/copy commands. Compare all current
16-bit words before expanding changed content for upload; retain raw-pointer
freshness, stride, GDI completion, ownership exclusions and config-off behavior.
No injected source, hook, signature, supported-build address or CSV changes.
`required_user_action: []`. The replacement live recording is present and strict
composition replay passes its bounded prefix. No further capture is needed to
validate this scoped optimization; overall gameplay speed remains unaccepted.

### Durable world jobs and fused native resource transactions

Existing dependencies: `c3x_renderer_native_image`,
`c3x_renderer_native_lifetime`, the installed native image ownership hooks,
`C3X_NATIVE_IMAGE_PRESENT`, and the existing GPU map/camera exports. Combined
content jobs own copied world observations; GPU image commands can accompany a
resource operation in one ordered worker transaction. These changes are inside
the renderer DLL. Native CPU access, original operation order, session retirement
and config-off behavior retain their existing contracts. No injected source,
patch signature, supported-build address or CSV entry changes.

`required_user_action: []` for this DLL-only change. If the preceding camera
consolidation bridge changes have not been installed, run `INSTALL.bat` before
the next game test. Staging must preserve the matching qualified short-capture
receipt; no new manual recording is required for these automated comparisons.
