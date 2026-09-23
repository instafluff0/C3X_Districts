# Renderer roadmap and current status

This is the short current-work entry point. The
[Renderer64 scene and motion contract](renderer64_scene_and_motion.md) defines
the adopted architecture and UX behavior. The
[64-bit migration plan](helper64_migration_plan.md) records the process
boundary and earlier gate evidence. Detailed graphics, source and historical
findings remain linked from [the documentation index](README.md); Git holds
the former 1,326-line roadmap.

## Current state

- Milestones 1 and 2 and M3.7 have automated acceptance evidence. M3.8–M3.12
  and the integrated M3.13 acceptance are not complete. Existing x86 gameplay,
  native UI, visibility and config-off contracts remain authoritative.
- The production renderer already retains world content, copied scene inputs,
  unit instances, GPU-ready geometry and independent visual samples. Shore
  waves, water motion and reflections are on for normal validation. Eligible
  selected/work units animate; unselected idle and fogged content stay frozen.
- Gate 1 proved a Windows cross-process graphics boundary. Gate 2 proved
  real-core scene/pixel parity and a controlled x86 address-space advantage.
  Neither proved live gameplay FPS or a device-loss fix.
- An opt-in cross-process surface candidate exists in the current checkout.
  It can display real frames without a per-frame shared-image return to Civ III.
  It is not the installed default. Its strict replay reaches a native handoff
  where the recording lacks an external native input; actual game UI ordering,
  autonomous Renderer64 cadence and complete lifecycle acceptance remain open.
  Earlier paired speed/pixel numbers accidentally compared the shared-image
  route with itself and are invalid for this surface.
- The unit body now carries a separate copied native visual observation.
  Renderer64 can smooth movement between successive accepted pixel samples;
  full event-scoped A-to-B timing and selection/path alignment are still open.
  Off-screen tile/city state starts from copied pages and receives bounded
  notifications at existing move, city, improvement and worker transitions.
  Other transitions rely on the post-interturn recovery pass; tile records
  are not a complete unit roster.
- World-page capture stops after one map/viewer snapshot; explored tiles carry
  full appearance and never-explored tiles carry only visibility and terrain
  topology. Background art preparation skips never-explored cores. Accepted
  `Unit_move` publishes a bounded old/new neighborhood and a stable-ID move
  record for visible endpoints, including entry from a hidden tile. The move
  record retires the prior pixel prediction; a following native body observation
  supplies the precise screen anchor. The existing `Leader_spawn_unit` hook
  publishes a scoped stable-ID birth; unit draw copies action/HP, and
  `Unit_despawn` retires that identity. All four event kinds cross the same
  ordered Renderer64 IPC and replay stream. Retirement prevents a later stale
  observation from reviving the ID until a new birth. Repeated unchanged
  unit facts are coalesced before cross-process delivery. Visible city/improvement/worker changes
  request a 13-tile local connectivity neighborhood, while off-screen changes do not dirty the
  native Animator. An exact native view redraw is still requested for on-screen
  changes and movement fog. Hidden tile deltas update fog status while retaining
  the last visible content until reveal. Full directed segment timing and a proved
  hidden-to-visible animation are still outstanding. A post-interturn audit re-arms
  one paged full-world reconciliation for missed off-screen changes; a rejected
  sparse change also requests that audit at the next native view. Synthetic
  reveal and wrap tests pass; live first-move reveal and input-to-displayed-frame
  latency are not yet accepted.
  The Standard-sized 12,800-tile synthetic snapshot is 100 pages once per
  scope or post-interturn recovery; a move inspects at most 170 candidate tiles
  and copies only changed visibility records. The measured synthetic
  12,800-tile capture and publication copy took 2.689 ms total on the Mac;
  region leases took 15.703 ms. These are workload bounds and a standalone
  CPU measurement, not a measured gameplay FPS improvement.
- Verification for this cutover: 308 production contract tests passed (two
  skipped), the approved injected compile passed, day/night unit behavior
  scenes passed, and an x86-to-Renderer64 direct-surface roundtrip accepted
  ordered birth, move, action/HP, retirement, stale-viewer rejection and ID
  reuse. An identical subsequent state caused no extra helper IPC call. No
  new CSV patch symbol was needed. Civ III gameplay was not launched by these
  checks, so first-move reveal timing and actual FPS remain unmeasured here.
  The unit warm/cold witness also records the observed sparse, at-most-five-level
  color variance in top-edge grass pixels; larger channel differences still fail.

## Architecture cutover before FPS tuning

1. **Authoritative state delivery.** Civ III/C3X publishes a scoped snapshot
   and ordered, versioned changes for tiles, cities, visibility, unit lifecycle,
   actions, selection/path and camera. The game thread alone reads game objects.
   Keep injected hooks small; value copying, diffing, sequencing, recovery and
   storage belong in Renderer. Reuse the existing publication journal and
   page capture for initialization/reconciliation rather than creating a second
   retained world. The scoped snapshot, move-neighborhood updates, local
   city/worker changes, unit birth/move/action-HP/retirement stream and bounded
   recovery are implemented. Next, close remaining tile/city transition gaps,
   prove every intermediate combat-HP revision at the right time, and deliver
   selection/path and camera notifications with integrated native/UI replay.
2. **Cross-process presentation.** Civ III owns its window and creates a
   surface per generation. Renderer64 owns the D3D scene, final image, visual
   clock and presentation to that surface. Remove per-frame Civ III visual
   requests and shared-image adoption from the normal path once the new route
   covers their callers. Keep the shared-image route as a controlled recovery
   path until native UI and device/window transitions are proven.
3. **Smooth directed actions.** Civ III decides legal moves and outcomes.
   Renderer64 samples accepted movement segments between authoritative anchors
   and runs visible ambient/work poses independently of the 66 ms game timer.
   Corrections, interruptions, fog, unit removal and camera jumps supersede
   stale visual segments. Selection and route markers stay aligned with the
   same sampled unit and view.
4. **Integrated correctness gate.** Focused contract tests precede one
   native/UI-interleaved replay and one staged, identified game build. The game
   check covers Scout travel, worker action, ground combat, bombardment,
   air combat/interception and health-bar timing,
   continuously moving visible water/resources during movement and interturn,
   selection/path alignment,
   fog/reveal, camera jumps, city/UI transitions, partial native transfers,
   reset and config-off. A missing recorded external input is a replay capture
   gap to repair at this gate, not a surface performance conclusion. A stable,
   usable game experience is required before broad FPS tuning.

## Then optimize the complete workload

Measure the playable build's input-to-correct-display latency, delivered visual
intervals, GPU work, native UI/composition cost, Renderer64 CPU/VRAM, Civ III
address-space use and process-transfer costs with all water effects enabled.
Use those measurements to improve pass submission, residency, batching,
frame pacing and arbitrary camera changes from every trigger. The Standard-map
goal remains under 33 ms p95 for coherent prepared navigation; Huge-map
capacity (about 12,800 actual tiles) is reported separately. Do not claim a
frame-rate or memory improvement merely because work moved to Renderer64.

Retire superseded x86 world preparation, raster caches, per-frame shared-image
handoff and duplicate scene owners only after the new route and config-off/
recovery behavior cover their callers. Preserve necessary native UI, replay,
partial-transfer and fallback contracts. M9 natural wonders, M10 constructed
wonders and M11 Districts remain deferred.

For detailed evidence, use [Gate 2](helper64_gate2_results.md),
[surface trial](direct_surface_trial.md),
[working-set results](frame_working_set_results.md) and
[recorded workload](recorded_renderer_workload.md). These are evidence, not
alternate work queues.
