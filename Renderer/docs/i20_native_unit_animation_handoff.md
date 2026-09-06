# I20 Native Unit Animation Handoff

## Decision

I20 must keep Civilization III's unit animator running as the authoritative
director and replace only the pixels of the vanilla FLC unit body. C3X must not
reimplement movement, combat sequencing, gameplay waits, damage, retreat,
death, victory, selection, visibility, stacking, or removal.

The intended production order is:

1. Civ III chooses the visible unit, action, direction, target, timing, and
   wrap-aware screen occurrence.
2. Civ III advances its normal `FLC_Animation` state, including invisible FLC
   frames after custom body ownership is enabled.
3. C3X captures an immutable unit record at the Animator-owned dynamic-plane
   draw boundary.
4. The renderer maps the native action and authoritative progress to the
   approved L20 model and clip, then draws at Civ III's current pixel anchor.
5. Civ III continues drawing its selection underlay, health/activity/status
   marks, stack indicators, and related unit HUD, and continues its normal
   gameplay wait and lifecycle code.

No renderer-owned simulation or independent animation loop is permitted.

Armies use the same rule as a composite instance, not as a special pre-baked
unit. Civ III's decompiled `field_1B0[1]` is the authoritative displayed-member
ID. Its dedicated draw path advances and draws the Army body beside that exact
member body, at audited 40-pixel normal and 20-pixel reduced-zoom horizontal
offsets. I20 captures both native action cursors and anchors, resolves the
member through the ordinary arbitrary-unit path, and resolves a separate
era-profiled Great General commander. It retains the parent Army HUD exactly
once. Full evidence and the offline contract are in
`army_rendering_strategy.md`.

## Confirmed Native Call Chain

The installed-source audit establishes the following useful behavior:

- `on_timer_0x9F6500` runs at approximately `0x42` ms and calls
  `Animator::refresh`. The existing C3X timer patch can mark the native Animator
  dirty before calling the original timer; it must never render or blit from
  the callback.
- `Animator::refresh` calls `Animator::update` when native camera, map, unit, or
  dirty state requires work.
- `Animator::update` rebuilds Civ III's visible unit list in native display
  order, advances animations, selects the correct wrapped camera origin through
  `Animator::FUN_004f0b90`, computes `Unit::FUN_005cbc30` dirty rectangles,
  calls `Unit::tick_anim` on `Units_Control.Data.Canvas`, and presents the
  unioned dynamic region.
- `Unit::tick_anim` performs visibility/city/stack checks, draws the native
  selected-unit underlay, executes one of two direct FLC body blits for normal
  or reduced zoom, and then draws retained health/activity/stack work.
- The FLC body does not pass through `Sprite_draw_on_map`; that existing hook
  cannot suppress a unit body.

Relevant decompiled source locations in `ref/Civ3Conquests_master.exe.c`:

- `on_timer_0x9F6500`: source line 167105.
- `Animator::refresh` / `Animator::update`: source lines 179231 and 179298.
- `Animator::tick_all_unit_anims`: source line 180614.
- `Unit::FUN_005cbc30`: source line 336591.
- `Unit::tick_anim`: source line 336702; normal/reduced body calls are currently
  visible at source lines 336845-336868.

Decompiler line numbers are navigation aids, not supported-build addresses.

## Authoritative Unit Record

Capture the following directly from Civ III at the dynamic draw boundary:

| Responsibility | Native source | Renderer use |
|---|---|---|
| Identity | `Unit.Body.ID`, type, civilization/owner | Stable instance/model identity; ownership is not assumed to be the displayed color |
| Display color | Native unit-body display-civ decision, then `leaders[display_civ_id].Color_Table_ID` | One 0..31 palette-row selector per instance; preserves hidden-nationality/viewer behavior |
| Effective palettes | `units_image_data.Color_Tables[0..31].JGL_Color_Table` after game/scenario load | One 64x32 RGBA8 LUT, including partial scenario `ntpXX.pcx` overrides |
| Action | `AnimationSummary.current_anim_type` and `queued_anim_type` | Default/fidget/run/attack1-3/death/fortify/victory/build/worker clip mapping |
| Facing | `direction` and `direction_2` | Approved eight-direction pose/clip orientation |
| Placement | `pixel_loc_x/y`, wrapped origin selected by `Animator::FUN_004f0b90` | Exact normal/reduced-zoom body anchor |
| Motion target | `pixel_target_x/y`, tile coordinates | Validation and optional between-tick interpolation only |
| Progress | native frame/progress fields plus effective native clip duration | Normalize an approved 3D clip to Civ III's authoritative action duration |
| Visibility/order | Animator visible-unit list and native checks | Never reveal hidden units; preserve stack/display order |
| Retained HUD | current/active/selected, damage/hit points, native dirty rectangle | Alignment evidence only; Civ III remains the owner |

Use a stable event ID formed from unit ID, observed action-transition serial, and
the relevant movement/combat participant or segment identity. A repeated redraw
must not restart a clip. Loading, teleport, upgrade, capture, despawn, visibility
loss, renderer reset, or an incompatible native transition terminates or rebases
the presentation event.

Owner color does not create per-civilization models, textures, materials, or
clips. The converted material supplies a neutral base and a continuous
`civ_color_weight`; the current lab calibration conservatively modulates that
base instead of replacing it with a fully saturated palette shade. The unit
instance supplies `display_color_table_id`;
`palette_generation` identifies the current 64-by-32 lookup. Capture or
display-identity changes only update the selector. Loading a game/scenario
rebuilds the LUT from Civ III's already-resolved tables and increments the
palette generation.

Do not derive the row directly from `Unit.Body.CivID`. The native body path can
substitute civ 0 when another player sees a hidden-nationality unit. I20 must
reuse that visibility-conditioned display decision so custom units neither
reveal their actual owner nor disagree with retained native UI. The checked-in
`tools/asset_compiler/unit_owner_color_runtime.json` is the executable offline
handoff for this policy.

## Movement And Combat Synchronization

Ordinary movement already assigns a native pixel target, direction, and run
state and repeatedly enters `Animator::update` until the unit reaches the target.
C3X should draw at the current native `pixel_loc_x/y`. Higher-frequency visual
interpolation may later be evaluated between two captured native anchors, but it
must snap to native endpoints and may never move gameplay state.

`Fighter::begin` records attacker, defender, attack/defense direction, wrapped
locations, visibility, and whether animations should play. Its combat approach
calculates a point between the attacker and defender, assigns that native pixel
target, and waits through ordinary Animator refreshes until the attacker comes
to rest. `Fighter::fight` then queues fortify/attack states, applies Civ III's
combat results, queues death or victory, and eventually calls `Unit::despawn`.

I20 therefore observes and renders these native transitions; it does not script
an attacker walk-up, infer a hit, choose a death, or decide when combat ends.
The invisible native FLC continues advancing so existing gameplay waits and
completion conditions remain unchanged. A differently authored 3D clip is
normalized to the native action duration instead of extending or shortening it.

## Existing Patch Coverage

The following current `civ_prog_objects.csv` entries are sufficient for native
direction/state/event capture unless an I20 fixture proves otherwise:

- `on_timer_0x9F6500`: request ordinary Animator work while a visible custom
  animation needs another frame.
- `Unit_move` and `Unit_move_to_adjacent_tile`: movement intent, endpoints, and
  segment direction.
- `Fighter_begin`, `Fighter_animate_start_of_combat`, and `Fighter_fight`:
  participants, directions, combat-event boundaries, and outcomes.
- `Animator_play_one_shot_unit_animation`: fortify, attack-family helper calls,
  build, death, victory, and other one-shot transitions that reach it.
- `Main_Screen_Form_set_selected_unit`: selection transitions and retained
  underlay alignment.
- `Unit_despawn`: authoritative removal and event cleanup.
- Existing bombard hooks/calls: projectile source/target and bombard event
  boundaries for the later I20/I7.5 split.
- `Units_Image_Data.Color_Tables[0..31]`, `Leader.Color_Table_ID`, and
  `Unit.Body.CivID`: effective scenario palette contents and normal owner-color
  lookup. These are data reads, not new patch dependencies. The eventual body
  boundary must also expose or reproduce the native viewer-conditioned display
  civilization used for hidden-nationality rendering.

`Animator_update` and `Unit_play_attack_animation` are already callable for
inspection but are not entry-patchable. The current audit does not require
changing either capability.

## Unresolved Body-Only Patch Spike

At least one new patch boundary is likely necessary because no existing patch
can suppress the two direct FLC body blits while leaving the rest of
`Unit::tick_anim` intact. This is the only presently identified likely new Civ
III patch dependency for core I20 unit replacement.

Resolve it at the start of I20 in this order:

1. Disassemble all supported executables around `Unit::tick_anim` and confirm
   the normal-zoom and reduced-zoom body call sites, the selected-underlay call,
   and the post-body HUD call.
2. Prefer exact body-call replacements if they can identify the owning `Unit`
   and invoke the renderer without intercepting unrelated sprite draws.
3. Otherwise test a `Unit_tick_anim` inlead that establishes narrow current-unit
   context, paired with body-sprite interception guarded by exact canvas and
   exact current-frame sprite identity.
4. Reject a whole-function reimplementation unless the smaller boundaries are
   disproven; duplicating `Unit::tick_anim` would be brittle and risks losing
   native visibility, army, city, selection, and HUD behavior.
5. Prove configuration-off byte/pixel behavior, both zooms, ordinary units,
   armies, stacked units, units in cities, hidden-nationality units, movement,
   combat approach, fortify, attack, death, victory, and retained HUD ordering.
   Army proof must suppress both native child bodies atomically while retaining
   the parent HUD; an empty Army has only the commander, and an authoritative
   displayed-member change replaces only the custom member child.
6. Only then add a `required_user_action` entry to the dependency ledger and
   `project_status.json` with exact symbol names, patch capabilities,
   signatures, and GOG/Steam/PCGames.de addresses. Do not edit
   `civ_prog_objects.csv` automatically.

Likely audit candidates are `Unit_tick_anim` and the two direct body-draw call
sites/primitives currently decompiled as `Sprite::FUN_005f88b0` and
`Sprite::FUN_005f8940`. These are candidates, not requests: exact supported-build
addresses and the minimal safe combination have not yet been proven.

## Required I20 Evidence

- Native transition traces show identical unit/action/event IDs, actions,
  directions, pixel anchors, endpoints, and outcomes with custom rendering off
  and on.
- Timestamp/progress fixtures cover default, fidget, run, fortify, attack1-3,
  bombard, victory, death, capture, build/worker actions, interruption, and
  missing-clip hard failure.
- Movement and combat-approach fixtures cover both zooms, scrolling during the
  action, horizontal wrapping, clipping, skipped frames, focus/modal pauses,
  stack changes, and renderer reset.
- Civ III's native selection underlay and health/activity/status/stack overlays
  remain correctly aligned and appear exactly once around the custom body.
- Custom-on failure never restores the vanilla body. Civ III gameplay timing,
  damage, retreat, death, despawn, audio, and waits remain unchanged.
- Static/default units do not force a retained-map rebuild; dynamic dirty-region
  work remains bounded and the timer queue never grows.
