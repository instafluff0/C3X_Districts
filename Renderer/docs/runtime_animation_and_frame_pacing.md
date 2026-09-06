# Runtime Animation And Frame Pacing Contract

## Purpose

The custom renderer is a guest in Civilization III's existing UI, simulation, animation, and map-render loop. It must improve presentation without creating a second game loop, advancing gameplay state, blocking Civ III, or rendering from an unsafe callback.

This contract governs animated units, effects, environmental motion, selected-unit indicators, frame scheduling, and performance. It applies before M7.4/M7.5 and is implemented as M5.3 after the live scene-export bridge is proven.

## Loop Ownership

Civ III remains authoritative for:

- Simulation state, turn processing, unit positions, selected/current unit, combat outcomes, visibility, and removal.
- The start, transition, interruption, and completion of gameplay actions.
- Window messages, modal state, focus, map scrolling, viewport size, and requests to draw the map.
- Gameplay waits and animation pacing that affect when control returns to the player or AI.

C3X owns only presentation state derived from those facts:

- A monotonic presentation clock and stable animation/event identities.
- Interpolation between authoritative anchors.
- Animation clip selection, normalized phase, blending, particles, and renderer-only ambient motion.
- Dirty-state tracking and requests for another normal Civ III map redraw.

The renderer DLL never runs an independent presenter, message pump, simulation
loop, swap chain, or autonomous frame-producing thread. It may use one dedicated
renderer worker to consume an immutable snapshot and own D3D state. That worker
has a capacity-one, no-backlog handoff, never reads Civ III pointers, and never
presents. The current synchronous ABI waits for the exact submitted sequence;
Civ III's UI thread remains authoritative for capture, redraw decisions, and the
final serialized GDI copy into Civ III-owned surfaces. Renderer work may enter
through the retained map plane or, after M7.4 proves its hook, the Animator-owned
dynamic unit plane; both remain ordinary Civ III-driven updates rather than a
second game loop.

## Frame Scheduling

The default path is redraw-driven:

1. Civ III reaches its normal `Map_Renderer_m71_Draw_Tiles` path because gameplay, scrolling, UI, or static-map invalidation requires a map update.
2. C3X captures one immutable map snapshot containing authoritative state, anchors, viewport, environment, and one presentation timestamp.
3. The off-screen renderer evaluates map-owned animation against that timestamp, renders at most once for that map pass, and returns control for Civ III's retained map overlays and UI.
4. Static frames schedule no further work.
5. When visible continuous renderer-owned animation is active, C3X uses Civ III's existing timer/redraw mechanism only to request ordinary Animator work. The callback marks work dirty; it does not invoke D3D, capture the map, blit, wait, or recurse into drawing.

Civ III updates native unit bodies on `Units_Control` after the retained map work. M7.4 must capture and composite renderer-owned unit bodies in that dynamic plane, using Civ III's visible-unit order, wrap-aware animation anchor, and dirty rectangle. It must not force every unit tick through the terrain capture or bake moving units into the static map cache. See `docs/civ3_render_loop_viability.md`.

The I20 implementation handoff is `docs/i20_native_unit_animation_handoff.md`.
It requires the native FLC state machine to keep advancing invisibly while only
the body pixels are replaced, so Civ III continues to own movement targets,
combat approach, action transitions, waits, outcomes, and removal.

The existing `on_timer_0x9F6500` cadence is approximately 66 ms, or about 15 updates per second, and is already patched for C3X tile-animation scheduling. M5.3 must audit whether this is the safest redraw request source. The first implementation should match Civ III's native cadence. Optional 30/60 FPS modes may be considered only after profiling proves a safe higher-frequency UI-thread invalidation path; they are not release assumptions.

There is no `Sleep`, busy wait, render-until-caught-up loop, or fixed simulation step inside a Civ III draw call. Missed presentation frames are skipped rather than queued.

## Time Model

Runtime elapsed time uses `QueryPerformanceCounter` and one cached frequency. Animation phase derives from an absolute epoch or authoritative progress:

```text
phase = clamp_or_wrap((presentation_time - start_time) * playback_rate / duration)
```

It is not accumulated from the number of rendered frames. A slow frame therefore reduces visual smoothness without slowing movement, extending combat, or changing game state.

Every frame snapshot records a monotonic presentation timestamp. Replayable fixtures replace it with an explicit deterministic time/progress value; offline tests never depend on wall-clock time. Large discontinuities caused by loading, focus loss, modal dialogs, or debugger pauses must be clamped or rebased according to the action's pause policy.

## Dirty And Idle Policy

C3X requests another frame only while at least one visible responsibility requires it:

- A unit is moving, attacking, fortifying, dying, or blending between clips.
- A selected-unit underlay or other enabled map indicator is animated.
- A visible projectile, particle, tile effect, water/material effect, or smooth day/night transition is active.
- The camera, viewport, visibility, environment, asset mapping, or authoritative map state changed.

Otherwise the renderer sleeps naturally because Civ III does not call it. Hidden, minimized, loading, modal, and non-map states do not schedule continuous renderer updates. Off-screen and fog-hidden objects do not keep the map hot unless Civ III requires their animation timing for gameplay.

Dirty state is split so later optimization can avoid unnecessary work:

- Scene/camera dirty: recapture and rebuild visible instances.
- Static-map dirty: rerender terrain, water, features, infrastructure, resources, and cities as owned.
- Dynamic dirty: rerender units, selection underlays, projectiles, and effects.
- Composite dirty: recopy/recompose cached outputs when the destination changes.

Correct whole-frame rendering comes first. Cached static/dynamic passes, GPU-resident composition, dirty rectangles, and asynchronous readback are later optimizations that must preserve the same frame contract. Dirty bits are hints only: every reuse decision also compares a content-derived fingerprint, so a camera/zoom/wrap/clip or scene change cannot be misclassified as an animation-only redraw.

## Unit State Machine

Each captured unit instance has a stable unit ID and includes:

- Unit type, owner, class, era, hit points/damage, visibility, and stack/display order.
- Current tile and authoritative current draw anchor.
- Facing using Civ III's eight directions.
- Action state such as idle/default, fidget, run, fortify, fortified, attack variants, bombard, victory, death, capture, worker actions, load/unload, paradrop, or scenario-defined action.
- Stable action/event ID, start timestamp, duration, playback rate, normalized progress, loop mode, and completion policy.
- Movement start/end map coordinates and pixel anchors, path segment identity, and interpolation progress.
- Attack target unit/tile and target anchor when applicable.
- Current/active/selected flags and the anchors needed to keep Civ III-owned selection, health, activity, status, and stack overlays aligned with the replacement body.

State transitions are edge-triggered by Civ III events or observed authoritative state changes. A stable event ID prevents an attack or death clip from restarting on every redraw. The renderer may blend clips visually but never invent a move, hit, death, target, or completion.

Movement interpolates between captured Civ III anchors. It snaps to the new authoritative anchor after completion or mismatch, map scrolling shifts both endpoints consistently, and wrapping paths use Civ III's chosen on-screen representation rather than a guessed shortest path.

Attack, victory, and death timing initially mirrors Civ III's effective animation duration. Replacing a vanilla unit animation must neither shorten nor extend a gameplay wait until a specifically audited synchronization hook can signal equivalent completion. A unit is removed only when Civ III removes it; a renderer-owned death pose may persist only within an explicitly captured presentation event.

Interruption rules cover movement-to-combat, combat-to-death, fortify/unfortify, selection changes, stack cycling, visibility/fog changes, teleport/upgrade/capture, loading a save, and renderer reset. Once I20 transfers unit-body ownership, missing clips fail the custom unit pass visibly and never restore the vanilla unit body.

## Native Unit HUD Ownership

M7.4 replaces only the vanilla unit body sprite. Civ III remains the initial and required owner of the current-unit cursor/ring, green health bar, left-side activity/status marks, stack indicators, and related unit HUD. Suppressing the vanilla body must not suppress these draws, and the renderer must not duplicate, relight, recolor, or occlude them.

The retained native overlays must follow the same authoritative unit anchor used by the 3D body during movement, map scrolling, clipping, combat, stack cycling, and selection changes. If a model, clip, device, or bridge operation fails, the custom body pass reports a visible hard failure without restoring the vanilla body; the retained overlays remain outside body ownership. Transferring any of these overlays to C3X later is separate work requiring explicit approval, a versioned ownership contract, and its own ownership and visual gate.

## Selected-Unit Underlay

The current-unit indicator is an independent instance, not baked into every unit model. It records:

- Target unit ID, current/selected state, visibility, and anchor.
- Stable animation epoch/phase, color or owner relationship, scale, and opacity.
- Depth role: above terrain and tile-bound art, below the target unit model and its health/status overlays.
- Ownership mode: `civ3` for M7.4.

It remains a retained Civ III layer throughout M7.4. The scene records enough state to test its anchor and depth independently, but renderer-owned `augment` or `replace` behavior is deferred until the user explicitly reopens that scope.

## Effects And Synchronization

Projectiles and transient effects use stable event IDs, source/target anchors, spawn time, duration, phase, and cleanup policy. They follow Civ III combat events and outcomes. M7.5 owns projectile flight, impacts, particles, interruption, and cleanup; M7.4 owns the unit clip and pose participating in that event.

The detailed bombard, bombing, Fire Rate, water-impact, audio-ownership, and
nuclear handoff is frozen in `bombardment_and_explosion_effects.md`; its
executable offline contract is
`tools/asset_compiler/combat_effect_contract.json`.

Ambient attachments use the same scheduler contract. Static emissive materials and static analytic lights invalidate only when captured environment or owning-object state changes. Visible flames, smoke, steam, animated water, flickering lights, or moving beacon effects contribute stable animation IDs and request bounded frames; daylight-disabled, hidden, off-screen, paused, fallback, and completed attachments contribute none. Their phase is derived from absolute presentation time plus a stable seed, never accumulated by rendered-frame count.

Audio remains Civ III-owned initially. Any later synchronization work must avoid playing both vanilla and replacement sounds or changing gameplay waits.

## Performance Policy

Frame-rate goals are measured on representative saves at supported viewport sizes, not assumed from standalone rendering:

- No active animation: zero renderer-requested redraws; ordinary Civ III redraw behavior is unchanged.
- Initial animated target: sustain Civ III's native approximately 15 Hz cadence without queue growth or input starvation.
- Static frame work is bounded to the visible map rectangle and one render per Civ III map pass. Dynamic unit work is bounded to Civ III's visible-unit list and affected dirty region.
- When over budget, skip presentation frames and reduce optional effects before reducing input responsiveness or altering simulation timing.
- Track CPU capture time, renderer CPU time, GPU render time where available, readback/blit time, total map-pass time, requested/presented/skipped frames, and visible instance counts.

The first bridge's synchronous readback is acceptable for correctness but not presumed final. Profiling decides whether GPU-resident composition, cached layers, dirty rectangles, reduced-resolution effects, or asynchronous readback are necessary.

## Verification Gate

M5.3 must prove:

- Static maps cause no continuous C3X redraw requests.
- Animated visible scenes request frames through Civ III's loop without reentrant drawing, a second message pump, `Sleep`, or busy waiting.
- Identical event traces produce identical poses at explicit timestamps regardless of rendered-frame count.
- Slow or skipped frames do not alter movement endpoints, combat results, action duration, unit removal, or turn timing.
- Loading, modal dialogs, focus loss, minimize/restore, scrolling, resize, and renderer reset do not create time jumps or stale animation events.
- Config-off behavior and Civ III-owned fallback remain unchanged.
- Frame-time telemetry demonstrates bounded work and no growing queue at the initial native cadence.

M7.4 adds direction, movement, clip-transition, stacking, interruption, death-lifecycle, and body-only replacement fixtures, including retained native selection/health/status HUD alignment and Z-order. M7.5 adds projectile/effect spawn, impact, timing, interruption, and cleanup fixtures.

## M5.3 Implemented Slice

M5.3 implements the scheduling boundary without prematurely taking ownership of an animated category. `Renderer/native/frame_scheduler.cpp` is a pure absolute-time decision function shared by the native smoke test and injected bridge. The injected `patch_on_timer_0x9F6500` supplies explicit QPC and UI-state inputs, records a single pending request, and marks Civ III's existing animator dirty before the original callback runs. In the currently implemented terrain slice, all capture, rendering, readback, and blitting still occur only inside `patch_Map_Renderer_m71_Draw_Tiles`; M7.4 adds a separately gated Animator-plane path for unit bodies.

The frame ABI now carries one immutable presentation timestamp, its QPC frequency, dirty bits, and a visible renderer-owned animation count. Renderer output echoes the visible count, reports whether continuous redraw is still needed, and includes native render/readback CPU ticks. Injection records bounded scalar maxima for capture, native render/readback, blit, and the full map pass, plus saturating requested/presented/skipped counters.

The current live renderer supplies zero visible animation responsibilities, so M5.3 changes no static-map cadence and does not duplicate Civ III-owned effects. Later category milestones activate continuous scheduling only after they can report a visible renderer-owned animation and satisfy their lifecycle/fallback gates.
