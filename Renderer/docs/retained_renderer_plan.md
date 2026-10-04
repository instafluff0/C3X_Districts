# Renderer roadmap and current status

This is the short current-work entry point. The
[Renderer64 scene and motion contract](renderer64_scene_and_motion.md) defines
the adopted architecture and UX behavior. The
[64-bit migration plan](helper64_migration_plan.md) records the process
boundary and earlier gate evidence. Detailed graphics, source and historical
findings remain linked from [the documentation index](README.md); Git holds
the former 1,326-line roadmap.

## Current state


- **October 4 cursor-edge, reveal and starting-shore fixes (live exploration passed).**
  The interrupted run `20261004-140536` failed when a 32×32 cursor background
  save crossed the source screen boundary, requested unsupported CPU readback,
  and stopped the GPU image session. Native copies now clip both source and
  destination, preserving untouched pixels. The actual JGL oracle fails before
  the fix and passes after; a fast 324-case regression covers edges and corners.
  Settled views also reject differently scaled cached terrain after a reveal.
  A second live witness exposed missing tree shadows while only two atlas pages
  per invocation were complete. Changed shadow pages now finish before shading
  receivers; unchanged pages still reuse exact contents. The production
  scheduling/table regression rejects the old behavior and covers 26 dirty-page
  subset sizes.
  Starting-shore waves were generated but misplaced and culled across the world
  seam. They now retain native screen anchors for each captured occurrence,
  canonicalize neighbor cells across either seam, and defer clipping until the
  current zoom is known. The old placement fails the wrapped-start regression;
  duplicate observations, repeated world copies and both seams pass afterward.
  Final capture `20261004-150753` completes three planned moves plus the released
  drag order, turn handoff, all four client edges and final scripted scrolling:
  2,120 sampled frames, zero native failures, unchanged source save. Starting
  waves draw immediately (three ribbons); after interturn, consecutive samples
  show moving foam and the selected worker's ring without additional input.
  The stationary-forest checker fails the old `20261004-142616` run (22.34 RGB
  excursion) and passes the final run (0.80; threshold 12). Sampled navigation
  passes: 1.75× settles in 0.283 seconds, maximum observed presentation pause
  0.157 seconds. These are helper observations, not physical scanout FPS.
  The transition category runs 215 tests with one optional skip; all executed
  tests pass. Six coastal-geometry/capture-checker checks pass separately.
  Matching-trio build, startup, staging and game-link hashes pass. No new
  injected hook or reference replacement was needed for cursor/shadow/wave fixes.
  Busy capture `20261004-151615` also passes with zero native failures and an
  unchanged save: every outward level through 0.5×, reversal, 3×/1× returns,
  scrolling and native Z. The four first outward transitions settle in
  1.62–2.01 seconds; longest sampled presentation pause is 0.816 seconds, with
  no observer gaps. This passes the coarse stall limits but is still noticeable
  heavy-map latency, not a claim of fully smooth zoom. The reported 40-second
  freeze was not reproduced. Wave motion/reflections remain enabled. Sampled
  views retain the map and aligned labels throughout the reviewed sequence.
  Evidence: `Renderer/.cache/navigation-quality-20261004/`.

- **October 4 transition sharpness and zoom capture.** The fallback image now
  uses native pixel resolution and prepares the outward destination scale once.
  A scene assembly/sort no longer discards all terrain pixels; bounded per-strip
  proofs preserve unchanged pixels while rejecting changed content, visibility
  and overlap order. Local repair checks ordering through removed contributors.
  World terrain, units and city labels use a stable capture envelope covering
  0.5×, so zoom alone does not retire the live map sampler. Native no-op camera
  clamps and pending same-view redraws keep the completed display identity;
  actual camera/projection/viewer changes retain their coherence barriers.
  No new patch-table entry, injected state or reference replacement is needed.
  The transition category passes 193 tests (one optional skip). Its production
  GPU checks preserve 442,368 one-pixel detail samples, reject all 12,288 samples
  in the deliberately blurred control, and exercise 114 water/depth cases.
  Five restored old behaviors fail the targeted regressions. The native compile
  smoke test, matching-trio startup and installation checks pass.
  `check_navigation_cadence.py` separately checks zoom completion and stalls;
  missing observer samples yield `incomplete`, never a performance pass.
  Evidence: `Renderer/.cache/navigation-quality-20261004/`. Live qualification
  remains in progress; the reported 40-second zoom stall has not been reproduced.

- **October 4 reveal and city transition coherence (staged for evaluation).**
  A changed scene could still select old terrain from the displayed zoom lane,
  the other cached lane, or the low-resolution bootstrap while water, rivers,
  borders and objects used current facts. Every preview now validates its
  geometry and visibility dependencies; invalid scene pixels cannot remain a
  preview, even during simultaneous lighting changes. Native camera/projection
  changes complete the map before UI painting. City presentation also holds
  intermediate transfers until its camera and native tile width match that map.
  No new hook, injected state, patch-table entry or reference image is needed.
  The transition suite passes (one optional skip); the new selection test checks
  512 intermediate color/depth states across four zooms, slow preparation,
  reveal/hide, city edits and cached-lane returns. Four deliberately restored
  old selection paths all fail. Five delayed native completions execute the
  full polling path, and 28 city-presentation cases cover bypass and release.
  The 663 retained GPU oracles, D3D visibility oracle, bridge/input checks and
  approved injected compilation pass. Disposable capture `20261004-122927`
  completes three moves, one research handoff and final scrolling; the revealed
  footprint has no black interior holes in the 24 checked arrival samples.
  Capture `20261004-124131` completes city founding, native 0.5×/1× and exit,
  with identical city anchors and no renderer errors. City panels remain present
  in all 310 sampled frames inside the city interval. Saves are unchanged,
  owned processes/tasks are closed, and installed trio/source hashes match.
  Evidence: `Renderer/.cache/visual-transitions-20261004/`. These are bounded
  sampled checks, not an all-frame or long-session guarantee. Cold city 0.5×
  preparation still takes roughly 3–4 seconds; this change preserves a coherent
  completed view during that delay rather than qualifying its responsiveness.

- **October 3–4 city interaction pressure and route previews (staged).** City
  build choices reproduced both multi-second queued UI work and a retained
  texture admission failure that stopped presentation. Image transport now
  copies the unchanged little-endian pixel payload in bulk. Small CPU UI edits
  upload only their changed rectangle, with periodic full replacement bounding
  retained patch history. The native image slot table accommodates city icon
  working sets without increasing its 64 MiB CPU pixel budget. Regenerable
  composition outputs can be reclaimed between and within frames, while pins
  protect current operands and targets; immutable uploads and retired views
  remain exact. The retained ceiling stays at 256 MiB. Capture
  `20261004-001156` completes all six build choices across 20 hover/select cycles,
  with no failures and 439 presentations in its final 9.1-second idle interval.
  Against the identical `20261003-234236` sequence, city image queue p95 falls
  from 744 to 21 ms and maximum from 1,582 to 99 ms; median is 1.1 ms. Large
  image batches during the 80–165-second city interval fall from 1,050 to zero.
  These are diagnostic queue times, not input-to-display latency. The source
  save is unchanged; test processes and the temporary task are closed.
  The 663 GPU pixel oracles, four full-resolution/lifetime pressure cases and
  29 focused host tests pass; the cache upload regression fails with the preceding
  frame-protection rule. Partial source updates pass 150 GPU edits, exact saved
  versions, failed-upload retries and the actual JGL adapter pixel suite,
  including 192-source eviction stress and 64 reusable city icons with zero
  repeat uploads across four cycles.
  Ten input-coverage tests and five backpressure tests also pass.
  The injected compile smoke test passes. The native
  route coordinate helper's half-world fold is corrected using captured map
  anchors: `20261003-225534` resolves y = -948 to the nearby y = 972 during
  600 held-pointer updates and completes movement plus an interturn.
  Evidence is under `Renderer/.cache/multi-turn-freeze/`. Existing patch
  symbols suffice; no reference images were replaced. These bounded checks
  do not certify an entire human play session.

  [`CAPTURE_GAMEPLAY.bat`](gameplay_profile.md) records ordinary user play
  without a replay journal. The separate collector saves window samples,
  memory, timing and bounded rolling logs while the game remains open or
  frozen. Renderer trace writers permit readers. Capture
  `20261004-063212-cbe2de` verifies independent shutdown with 285 window
  samples, 834 presentation records and no missing evidence; only the owned
  disposable game was subsequently closed.

- **October 3 idle animation after research (staged and tested).** The first
  turn with Enter accepting research reproduced a remaining freeze: native
  redraw ran, but the async bridge reused the old completed camera after world
  preparation retired its animation sampler. The bridge now invalidates that
  reuse while retaining completed pixels until a fresh camera is adopted.
  Capture `20261003-211547` has identical map pixels and no ring across 28 idle
  samples from 125–179 seconds. The same sequence in `20261003-213213` adopts a
  new camera at 116.36 seconds, shows the selected Worker's ring, and has 28
  distinct shoreline samples before the first subsequent input at 185 seconds.
  All three moves, one turn and final scroll complete without renderer errors;
  the disposable save is unchanged and owned game/helper processes are closed.
  Twenty-three focused host tests and the matching GPU async fixture pass.
  The regression fails against the previous bridge. The repeatable
  `research-turn` scenario supplements the later-turn tests below; those did
  not cover this first-turn handoff. Evidence is under
  `Renderer/.cache/turn-start-resume/`. No injected or patch-table changes.

- **October 3 unit redraw and layering follow-up (staged and tested).**
  The native Animator now consumes pending renderer redraws without requiring
  player input and recaptures cursor eligibility changes even when selection
  keeps the same unit. Own-unit exploration admits the accepted travel segment
  before destination sight catches up, while hidden foreign movement remains
  excluded. Territory finishes before main unit bodies; shadows retain terrain
  depth and bodies use an independent depth pass. Host regressions reproduce
  both missing native updates against the previous injected code. The D3D11
  overlap oracle and matching async bridge fixture pass, including unit
  self-depth, reflected occlusion, fog, pose transitions and 32 camera adoptions.
  The 135-second disposable `unit-turn` run shows both Scout bodies travelling
  through intermediate positions, the new-turn ring before further input, and
  aligned final scrolling with no native errors. The original save is unchanged.
  The units category suite passes 162 tests with one skip; 32 focused host tests,
  the depth oracle and injected compile smoke test also pass. These are bounded
  checks, not long-session certification.
  Evidence is under `Renderer/.cache/selection-motion-layering/`. Existing patch
  symbols suffice; no patch-table changes or reference replacements are needed.

- **October 3 interturn freeze repair (staged and tested).** Required world
  paging now waits for the renderer's data locks instead of using an
  opportunistic background query. The previous build reproduced an interturn
  `PENDING` result that permanently disabled native renderer integration while
  the HUD continued moving. The same disposable-save exploration/turn/scroll
  sequence now completes two moves, two world preparations and 32 camera steps
  without errors. Window evidence shows revealed terrain, automatic selection
  before further input, an idle selected worker and aligned scrolling; the
  original save is unchanged. Forty-six focused tests and the async GPU fixture
  pass. The new busy-worker regression fails against the old query. Evidence is
  under `Renderer/.cache/scroll-sight-turn/`; the repeatable `turn-scroll` scenario
  is documented in [scripted testing](../tools/scripted_game_test.md). No injected
  source or patch-table changes were needed. This bounded check does not certify
  long-session performance.

- **October 3 performance overhaul (untested build).** The retained static
  layer now previews (whole-pixel shift, zoom resample, low-resolution jump
  bootstrap) and refines full quality under a per-frame budget instead of
  redrawing synchronously; shadows and city lights use a stable region of
  interest; ambient frames are vsync-paced on the DXGI latency signal. See
  [the overhaul note](performance_overhaul_20261003.md) for causes, switches
  that restore the old paths, and the game test plan.

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
