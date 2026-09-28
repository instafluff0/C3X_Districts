# Renderer roadmap and current status

This is the short current-work entry point. The
[Renderer64 scene and motion contract](renderer64_scene_and_motion.md) defines
the adopted architecture and UX behavior. The
[64-bit migration plan](helper64_migration_plan.md) records the process
boundary and earlier gate evidence. Detailed graphics, source and historical
findings remain linked from [the documentation index](README.md); Git holds
the former 1,326-line roadmap.

## Current state

- **Current priority: held-mouse targeting, then wheel zoom, easing and navigation performance.**
  Explosions and smoke are on hold at the user's request. The imported local
  effect assets remain available for later work.
  Candidate `c704bcc8bce5401294b88494fde7b312` is staged. Its async fixture
  passes at 60.00 FPS (59.95 during pose changes), with 32 adopted cameras,
  120 frames during a host pause, 73 publications during a renderer pause,
  p95 submission 0.925 ms and zero CPU map readbacks. These are fixture
  measurements, not live-game FPS.

  Small HUD background reads previously rebuilt full-screen source textures.
  Retained reads now preserve their actual read footprint and reuse a source
  view when it covers every needed pixel. Sparse and aliased inputs retain
  exact assembly. The GPU regression checks 32 small HUD reads over five
  changing 2240×1260 frames, with exact pixels and 2,920,704 assembled pixels
  per frame. Paired native transfers share immutable versions; keyed form
  transfers skip proven empty margins. No CPU map fallback was added.

  At one-Hz window sampling, the profiled held-drag rate improved from
  31.66 to 44.09 visual submissions/sec (`231750` → `232845`). Median sampled
  composition time fell from 29.976 to 17.680 ms. A DXGI presentation permit
  now bounds the display queue; its zero-timeout poll leaves command delivery
  available when the compositor is busy. It retains admission across static
  no-op frames. Matched one-Hz counter runs before/after that limit record
  42.74/42.46 presentations/sec in the same held-drag phase: the queue limit
  does not establish an FPS improvement.

  Capture `20260927-235859` bounds two destination-marker changes at 4–86 ms
  and 62–180 ms using compositor source timestamps. This is a bounded mouse
  witness, not a frame-perfect latency measurement or complete UX acceptance.
  All seven input commands complete, picks remain consistent, and no native
  failure, early exit or save change occurs.

  Navigation remains below the sandbox target. `20260928-000807` completes
  all 32 camera moves with the terrain, units and HUD visible in sampled
  frames. It records about 32.6 presentations/sec during scrolling and 46.1
  afterward. `20260928-001129` completes all ten wheel inputs at about 38.5/sec
  during the input sequence and 47.5 afterward. Both use one-Hz capture and
  the read-only presentation counter, with no renderer errors or save changes.
  These counters do not measure physical scanout. The roughly 52 FPS target
  is not yet met in these navigation workloads.

  Wheel activation was explicitly authorized and installed: the GOG m25 row
  changes from `ignore` to `repl vptr`. Thirteen zoom/input tests and the injected
  smoke passed at activation. Live `20260927-233038` delivers ten wheel events,
  including a 162 ms reversal and three accumulated 40-unit deltas. The center
  tile stays fixed and a round trip restores the original transform. There are
  no native failures, early exit or save changes. Smooth zoom remains pending:
  renderer-owned intermediate views must carry map HUD anchors and inverse
  picking together, while fixed UI remains in screen coordinates.

  A GPU completion gate never became busy and was removed. Removing the
  mid-frame flush moved its wait to later work and did not itself establish a
  speedup. A direct-to-retained-output experiment passed pixel checks but did
  not establish a performance benefit and was discarded.
  An opaque native GPU-copy trial also passed exact pixels but left matched
  scrolling cadence unchanged (32.57 → 32.56/sec). It was removed, and the
  source-matched `c704` trio was restored with all three hashes verified.
  Final focused verification: 35 tests completed (one optional executable
  audit skipped); the retained compositor has 126 exact GPU oracles. Injected
  sources are unchanged in this iteration, so reinjection is unnecessary.
  The [scripted testing guide](../tools/scripted_game_test.md) records mouse
  timestamps, buffered profiling, presentation counters and visible-latency checks.

- **Earlier movement evidence: tile travel and combat presentation.**
  Asynchronous fixture `94050b8afeb4467f89518784e5ed8391` passes at **59.26 FPS**,
  with 32 camera adoptions, 120 frames while the host pauses, 79 publications
  while the renderer pauses and zero CPU map readbacks. Submission p95 is
  0.793 ms, maximum 7.451 ms. These are fixture measurements, not live-game FPS.
  The matching bridge, x64 DLL and helper are staged together. Preserve the
  roughly 52 FPS sandbox baseline; further FPS tuning is deferred.

  The accepted native tile target starts a Renderer64-owned visual segment.
  Its clock begins on first rendering; native completion cannot truncate it.
  Neighboring steps preserve run phase. Run timing comes from the generic
  compiled clip header during asset loading. Bodies and selection rings share
  the copied scene's tile centers and visual clock. Native intermediate pixels
  no longer position the actor. The scoped GOG hook, config-off behavior and
  other-build limitations are in the [patch ledger](civ3_patch_dependency_ledger.md).

  Capture `20260927-183538` on that movement build shows both units at intermediate run positions and
  then the destination, including the first move's terrain reveal. The earlier
  source-pose rewind and frozen second move are absent in these samples. Native
  transfers retain the latest completed image during a camera handoff. Travel
  pauses for genuine camera changes; same-camera world updates keep sampling
  immediately. All 46 focused tests pass. The retained-composition GPU regression
  also proves repeated native transfers preserve the final pose after freezing
  its callback. Ten commands and three map-text events completed with no logged
  native failure or early exit, and the original save stayed unchanged.
  This is a bounded movement witness, not full gameplay acceptance: broader combat,
  native unit-status placement and complete UI coverage remain open.

  **Melee combat and presentation handoff now have a live witness.**
  Capture `20260927-183323` shows the attacker advancing to its accepted
  half-tile combat position, attack playback, death at that position, retirement
  and return to normal unit selection. Native combat completed, the original
  save stayed unchanged, and no renderer failures were logged. Native Warrior
  attack/death duration is 15 × 0.083 = 1.245 seconds. Repeated native captures
  no longer restart the copied action clock.

  The earlier fog-edge weapon clipping and stale source-stack civilian are
  absent in this capture. The existing GPU body draw marks depth-tested stencil
  coverage for the final fog pass. Native display-parent IDs choose the current
  stationary stack group, preserving grouped army bodies and admitted travel.
  Forty-six focused tests, six visibility tests, the injected smoke and native
  build pass. GPU coverage has 8,787,552 pixel checks; retained composition has
  126 exact checks. No new patch-table address or CPU rendering fallback.

  **Combat is not fully qualified.** Victory, retreat, ranged and army encounters
  still need live witnesses. Native health-marker placement and audio alignment
  remain separate checks. The user requested smooth limb transitions between
  actions; the current path still switches clips directly. The required local
  skeletal blending and binding metadata are documented in the scene/motion
  contract. Sampled window frames do not establish frame-perfect smoothness.

  Shared renderer assets now load from `patch_load_scenario()` after the C3X
  configuration. There is no separate progress form. Capture `20260927-162108`
  confirms native image loading happens before the camera/bounds are initialized;
  moving first-view preparation behind the existing loading bar remains open.
  Menu/reload validation is also open. Helper teardown now joins visual callbacks
  before unmapping their shared transport. The [scripted diagnostic](../tools/scripted_game_test.md)
  documents autonomous disposable-save tests and reports premature game exit.
- **Runtime boundaries.** Custom-on uses the matching 32-bit bridge, x64 DLL
  and helper, copied scene publication, retained GPU composition and an
  independent presentation clock. Native gameplay, pathfinding, turn logic and
  fixed UI remain authoritative. Config-off stays native. Water, shore waves
  and reflections remain enabled in normal validation. Licensed local assets
  and source findings are retained; M9–M11 remain deferred.
- Earlier migration experiments and superseded performance measurements live
  in Git and the evidence documents linked below. They are not current build
  receipts or proof of live-game performance.

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

**Deferred smooth edge scrolling.** Revisit this only after the sandbox's
representative scene, async camera and motion are working, and a playable
in-game integration has confirmed visual quality, map UI ordering, picking and
frame pacing. Then prototype a renderer-owned continuous display camera with
edge-distance speed and eased starts/stops. Civ III retains the committed camera,
wrap/clamp and gameplay authority; C3X intercepts only qualified manual pan
steps, while native recentering, jumps, zoom and config-off retain their existing
paths. Keep map-anchored overlays and clicks tied to the actually displayed
camera. Validate edge approach/release/reversal, diagonal motion, wrap/clamp,
zoom, jumps, clicks during motion and host stalls, with units and water animating.
Measure displayed frame intervals and input-to-display latency before promotion.

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
