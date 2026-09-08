# Native unit rendering contract

Civ III remains the animation director. The current renderer replaces eligible
FLC body pixels on the Animator-owned unit canvas; it does not replace the
unit routine, advance gameplay, infer combat outcomes or create a second loop.
The current visual pack and category checks are documented in
`Renderer/native/environment_refresh/UNIT_FIDELITY.md`.

## Current boundary

`patch_Unit_tick_anim` in `injected_code.c` establishes a scoped unit/canvas
context, calls the complete native routine, and restores the previous context.
The two guarded Sprite inleads consider only the exact current-frame body Sprite
on that canvas. Normal native visibility, selected underlay, FLC advancement,
movement/combat waits, health/activity/status marks and stack HUD remain intact.
Unrelated Sprite calls pass through unchanged.

The record passed to the DLL contains the native unit ID and Civilipedia key,
current/queued action, `direction_2`, `field_FC` cursor, the current action's
`Frame_Counts` length, body anchor, native Sprite dimensions, zoom, hour and
season. It is not reconstructed from a terrain traversal or an independent clock.
The effective palette comes from the actual native body call; the bridge reads
color index 6 and forwards its RGB. It does not assume that the owner is the
displayed civilization or implement the older proposed 64-by-32 palette LUT.

The bridge forwards both canvas and background DCs. The DLL resolves partial
coverage against the native underlay where the canvas is magenta-keyed.
Missing inputs, unsupported models/actions, unrelated canvases and rejected
pre-transfer draws retain the original body call. Configuration-off also retains
native rendering. This separate unit fallback is intentional; it does not permit
native terrain replay on the exclusive custom map plane.

On successful replacement, the bridge expands the display owner's existing
`Unit.Body.Rect` to contain the constrained body/shadow canvas. Native Animator
code unions that same rectangle after `tick_anim`, so presentation and later
erasure include the custom footprint. Failure leaves the rectangle unchanged.
The reduced hook replaces only scale `1, 1, 2`; other scales pass through.

## Armies and compound bodies

The current bridge distinguishes the commander Sprite from the representative
member Sprite and resolves the latter through `army_top_defender_id`.
Each body's own native cursor, type and anchor feed the ordinary renderer.
Dirty bounds expand the Army display owner's rectangle. Civ III retains the
parent HUD and member selection.

This implementation makes a replacement/fallback decision for each intercepted
body. It does **not** implement the older proposed all-or-nothing commander/member
preflight. Do not describe that proposal as current behavior or change the
accepted fallback during reorganization. `army_rendering_strategy.md` retains
broader composition requirements and source findings. A compound asset's parts
remain one generic model kit; they do not drive gameplay or Army membership.

## Native source evidence

The timer callback reaches `Animator::refresh`; `Animator::update` rebuilds the
visible unit list, selects the wrap-aware origin, computes dirty rectangles and
calls `Unit::tick_anim` on `Units_Control.Data.Canvas`. The selected underlay
precedes the direct body blit; retained health/activity/stack work follows it.
Unit bodies do not pass through `Sprite_draw_on_map`.

Useful navigation points in `ref/Civ3Conquests_master.exe.c` are
`on_timer_0x9F6500`, `Animator::refresh`, `Animator::update`,
`Animator::tick_all_unit_anims`, `Unit::FUN_005cbc30`, `Unit::tick_anim` and
the Army helper `Unit::FUN_005cc430`. Decompiled line numbers are not executable
addresses. Native movement and `Fighter` combat transitions continue to choose
targets, approach, outcomes, death/victory, sound and removal. Worker action/tool
source findings remain in `worker_builder_animation_mapping.md`; do not claim
the current bridge captures additional job fields that are absent from its record.

## Audited GOG boundary

The installed-source audit used GOG executable SHA-256
`c85aa82062a0249ce87cafe31d4ed94d98b4f4ef5727939d7311d289eae52710`.

| Responsibility | GOG address |
| --- | --- |
| Unit tick context | `0x005CBF50` |
| Normal body primitive | `0x005F88B0` |
| Reduced body primitive | `0x005F8940` |
| Ordinary normal/reduced calls | `0x005CC33B` / `0x005CC3A1` |
| Ordinary retained HUD | `0x005CC41D` |
| Army helper | `0x005CC430` |
| Army normal commander/member calls | `0x005CC861` / `0x005CC8BC` |
| Army reduced commander/member calls | `0x005CC926` / `0x005CC98D` |
| Army retained HUD | `0x005CC9EB` |

The selected underlay uses the separate `FUN_005f84b0` routine. Native Army
member placement uses 40-pixel normal and 20-pixel reduced offsets.
The body primitives return with `RET 0x18` and `RET 0x24`; reduced zoom has
nine stack arguments, including its legacy draw argument and effective palette.

The three GOG rows are already inleads in `civ_prog_objects.csv`.
Their exact current signatures and other-build limitations belong in
`civ3_patch_dependency_ledger.md`. Steam/PCGames addresses remain unverified;
zero is not a patch address. No CSV edit is requested for the Lab workflow.

## Verification and limits

`Renderer/tools/audit_unit_hooks.py` audits native entry/call bytes and the
current CSV. `Renderer/native/test_unit_bridge.cpp` executes the actual extracted
bridge against mocks, including argument forwarding, nested context restoration,
canvas/DC failure cleanup, both zooms, fallback, dirty bounds and retained-layer
ordering. Windows x86 verifies the calling convention. Shader, pose, cache,
color-key, RGB555/RGB565, input-guard and unit-shadow checks cover their separate
DLL responsibilities.

Run category checks through `python3 Renderer/renderer.py test units`;
current-code integration adds delivery behavior checks. The nine-family native
matrix is not the full 78-unit roster, and neither it nor source-payload parity
proves a live Civ III session. The category catalog describes current scope;
fixed references and disposable receipts are not release status. Only an actual
game check can establish observed in-game behavior.
