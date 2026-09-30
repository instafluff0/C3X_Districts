# Renderer roadmap and current status

This is the short current-work entry point. The
[Renderer64 scene and motion contract](renderer64_scene_and_motion.md) defines
the adopted architecture and UX behavior. The
[64-bit migration plan](helper64_migration_plan.md) records the process
boundary and earlier gate evidence. Detailed graphics, source and historical
findings remain linked from [the documentation index](README.md); Git holds
the former 1,326-line roadmap.

## Current state

- **Current focus: 50–60 FPS during idle, scrolling, zoom and map jumps.**
  The September 29 startup and camera-history fixes are staged. Native camera
  centering is preserved, speculative startup map draws are removed, and retired
  views release their rendering history. Both camera paths pass the 80-replacement
  GPU pixel/memory regression; the full retained compositor and projection checks
  pass. At the user's request, the final longer live freeze runs were cancelled,
  so sustained live qualification of the complete fix remains pending.
  Performance work uses preserved controls and standalone full-resolution
  measurements first, with water, waves, reflections and detail unchanged.


- **Previous graphics-quality work.** The user authorized completing the
  quality changes without intermediate approval checkpoints. Shared geometry
  now rasterizes at displayed zoom, with modest scene-wide CAS before native
  HUD and pose-local GPU self shadows for every unit. Sampling comparisons keep
  the native single-sample target; MSAA2 costs more during view changes. The
  [quality note](render_quality.md) records source findings, tests, captures and
  performance limits. Candidate `b71c8cee78144743bd2aac3ac7c7de4f` is staged;
  its async fixture measures 60 FPS and zero CPU map readbacks. Corrected live
  zoom capture `20260928-094902` measures 51.1 FPS at normal zoom, 35.1 over
  repeated zoom changes, and 49.5 at settled 1.25x. The final scroll control
  measures 44.6 FPS while moving and 50.5 afterward. HUD and city checks retain
  fixed labels and both native city zoom levels with the same center. The cold
  city-view preparation delay remains unresolved. These initial-save results
  do not establish uniform frame pacing or Civ VI visual parity. No fixed
  reference image has been replaced.


- **Previous functional baseline: smooth zoom.**
  Explosions and smoke remain on hold. Native Civ III HUD keeps its pixel size;
  city labels, status icons and map messages move with their attachment. Only
  Renderer64 world graphics scale. Fixed panels retain their screen position.

  The renderer now receives a copied zoom target through the existing async
  image queue. Canonical capture stays at 128 pixels; no camera recapture or
  redraw is requested for intermediate views. Renderer64 samples one damped
  view clock, and picking reads the scale of the last successful presentation.
  The helper wire is version 11; stage the bridge, x64 DLL and helper together.
  See [implementation and tests](custom_rendering_zoom.md).

  Live HUD capture `20260928-080539` completes both camera moves, three text
  events, native city zoom and advisor open/close without renderer errors or
  save changes. Sampled frames show fixed-size city labels at the new map
  positions, no old-label copies and preserved fixed panels. Retained UI
  backgrounds reference the current world/HUD selection; native UI commits
  retain their actual dirty rectangles. GPU regressions reproduce the old
  label and partial-panel failures and check the replacement pixels exactly.

  City capture `20260928-082829` verifies native 64/128-pixel zoom with an
  unchanged city anchor at both levels. The terrain is visible at both levels;
  wheel input and attempts to scroll at either screen edge leave the city
  centered. The native spotlight lifetime excludes custom zoom and manual
  panning, including the first city draw. The existing native centering helper
  now preserves both tile-row parity and the same vertical pixel offset.

  Evaluation candidate `248f18290d534006846bf930edd62dbe` is staged and console
  installed. Its async fixture passes at 59.89 FPS (59.75 during pose changes),
  with 32 adopted cameras, 121 frames during a host pause, 71 publications
  during a renderer pause, p95 submission 0.894 ms and zero CPU map readbacks.
  These are fixture measurements, not live-game FPS. Twenty-four focused tests,
  the retained GPU suite, injected compilation and console installation pass.

  Live performance remains open. One-Hz scroll control `20260928-083716`
  measures 31.7 presentations/sec during scrolling and 40.7 afterward, with
  all 32 camera steps completed and no renderer errors or save changes. The
  selected-world GPU copy reduction and bounded busy-transaction retry have
  not established a measured live FPS gain over the earlier 34.9/41.7 control.
  The roughly 52 FPS sandbox target is **not yet established in the live game**.
  Final zoom capture `20260928-083851` completes all ten wheel inputs with
  no renderer errors or save changes. Presentation telemetry records 68 distinct
  scales within 1.0–1.5, correct endpoints, reversal and three accumulated
  40-unit deltas. Reviewed window frames show aligned terrain, units and
  selection throughout, with fixed panels preserved. This is a functional zoom
  pass, not a claim of uniform frame pacing at the sandbox target. The user
  requested stopping once zoom passed, then authorized the graphics-quality
  work above. Cold city-view preparation and broader menu/reload work remain
  separate unresolved items.

  The first reduced city view still prepares new terrain geometry and can take
  about seven seconds. Centering and native zoom restrictions are verified;
  this cold preparation delay is unresolved. Earlier advisor queue exhaustion
  has not recurred in the recent HUD/advisor captures, but broader UI coverage
  remains open. No reference image acceptance is implied.

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

  **Startup follows the native camera and draw sequence.**
  `patch_load_scenario()` still blocks on the shared Renderer64 asset loader.
  Save cleanup, scenario placement and camera hooks no longer prepare extra
  maps or manipulate the loading bar. The first ordinary map draw completes
  its GPU request before native composition; later draws remain asynchronous.
  Capture uses an explicit native traversal, independent of debug-mode pass
  masks. Off-screen art preparation skips never-explored tiles while retaining
  their topology. The scripted diagnostic covers saved starts, new games and
  debug reveal/hide; see its guide for measured evidence and limitations.
  Helper teardown joins visual callbacks before unmapping shared transport.
  The [scripted diagnostic](../tools/scripted_game_test.md) documents disposable
  save tests, menu/reload checks and premature-exit reporting.
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
