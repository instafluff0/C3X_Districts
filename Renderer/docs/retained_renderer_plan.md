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
- `enable_custom_rendering = true` now selects Renderer64 as the sole custom
  backend, with the direct cross-process map surface requested in normal play.
  The 32-bit bridge, Renderer64 DLL and helper are built and staged together;
  `INSTALL.bat` has installed the matching injected game executable. An early
  JGL screen could previously instantiate the x86 renderer before Renderer64
  selection, silently forcing native fallback. Selection now precedes native
  screen tracking and repeat selection is safe. The exact installed-directory
  startup probe passes bridge selection, early screen-policy activity, definition
  loading and helper health. The user confirmed normal installation activates
  the custom renderer and reported mostly smooth idle play, but slow unit travel,
  scrolling and distant jumps. Complete UI ordering, autonomous cadence and
  lifecycle acceptance remain open. Earlier paired
  speed/pixel numbers accidentally compared the shared-image route with itself
  and are invalid for this surface. The Renderer64 short diagnostic is now
  source-matched and staged: its controlled direct-surface journal closed with
  7,137 calls and 87 accepted presentations, and exact/real-time same-route
  replays completed, with startup and launcher/window controls passing. These
  counts are capture/replay coverage, not a gameplay FPS measurement. The direct
  fixture still fails the legacy x86 pixel oracle,
  so it is not visual-parity evidence. A later 117-second live capture recorded
  317,273 calls and 554 presentations, but its journal and window evidence did
  not complete; PresentMon retained only one helper row. The pinned original
  replay stops at a missing exported-clock sample. The candidate derives that
  sample from the recorded clock return; explicit pixel-audit mode now allows
  implementation-only tile/animation counts to differ while preserving scene
  identity and ownership checks. The candidate then stops at a genuinely
  missing native external input before the first recorded move. This capture
  still cannot certify movement or displayed FPS across the architecture change.
  The 82 captured Renderer64 map requests measured 114.3 ms p95 including
  cold content; the 70 requests without object or reflection builds measured
  30.4 ms p95 inside Renderer64. Neither is input-to-displayed-frame latency.
  The capture also showed the x64 world-region cache disabled despite the
  game's enabled config: the helper had started before configuration. The
  Renderer bridge now sends current settings with definitions; an isolated
  early-start probe confirms the x64 cache and its related retention profile
  enable after late configuration. The corrected bridge and Renderer64 binaries
  are staged together; the resulting live navigation latency is not measured.
- The unit body now carries a separate copied native visual observation.
  Renderer64 samples a visible move continuously from its first body anchor
  to Civ III's accepted pixel target at one presentation pace for every unit;
  later sparse native poses do not change that pace. The retained draw area
  covers the complete segment. The x64 water predicate now admits visible
  real-terrain water and rivers even when the base terrain field looks like
  land. A live screenshot then showed a selected-unit ring with no unit body:
  retained replay shifted the draw into a local scratch image while the unit
  callback still used screen coordinates. The callback now rebases the body
  within that image; a D3D pixel test keeps the unit visible over two animated
  map samples. Focused unit/water tests and the units integration suite pass; live
  movement, continuous water and selection/path alignment remain unproven.
  Off-screen tile/city state starts from copied pages and receives bounded
  notifications at existing move, city, improvement and worker transitions.
  Other transitions rely on the post-interturn recovery pass; tile records
  are not a complete unit roster.
- A later missing-unit screenshot was captured before the corrected Renderer64
  trio was staged. The staged files match their built artifacts, the normal
  helper startup probe passes, and the retained unit pixel oracle passes. A
  subsequent live game result is still needed before claiming that every unit
  action is visible in play.
- Standard-size, 2240×1260, 100-destination readiness with all water effects on:
  world-scoped appearance validity first reduced geometry builds 2,507→242 and
  uploads 1,082→135 MB. The remaining 242 builds came from a view-scoped river
  node list changing prepared tile keys. The node list now derives from the
  copied full-world topology and refreshes only on world revision/scope changes;
  sorted spatial selection also bounds its camera cost. The same 100 destinations
  now make zero world compiler calls and zero geometry uploads. All six cold
  images are byte-identical to the preceding candidate, whose warm/cold oracles
  were already pixel-identical. The x64 full-readiness harness now fails if a
  prepared jump compiles or uploads geometry. Desktop median was 253 ms,
  p95 895 ms in this run; the long handoff
  stalls still violate the <33 ms goal, and the timing variation is not a proven
  displayed-speed gain. These changes are staged for a live game check. The
  bounded 768 MiB reflection cache is worthwhile:
  disabling it tripled reflected draws and raised desktop p95 to 1,024 ms.
- A 1.5 GiB reflection cache reduced reflected draws 92,988→52,502 and request
  p95 573→199 ms, but desktop p95 rose to 1,211 ms despite at least 5.68 GiB
  free physical memory. A serialized diagnostic found scene completion under
  20 ms p95 but one-pixel readback at 588 ms p95. Presentation phases place
  the remaining long wait in Renderer64's display-to-surface service; Civ III
  IPC/adoption and DXGI Present are fast. Removing Flush, reversing or omitting
  the surface copy, and adding a third swap-chain buffer did not improve the
  complete workload and were discarded. The harness disables autonomous visual
  frames, so ambient queue backlog is not this stall's cause. Keep the 768 MiB
  budget and investigate GPU/driver resource handoff while completing prepared
  world residency. All measured variants were no-stage trials.
- A full-resolution, Standard-map 100-destination phase trace now locates the
  long direct-surface wait in `OMSetRenderTargets` for the swap-chain target:
  552 ms p95 in one run, while the final draw and `Present` stayed below 2 ms.
  WDDM reported a 6.87 GB local budget and about 2.9 GB usage, so crossing its
  residency budget does not explain this fixture's stalls. At 1280×720, mean
  displayed latency fell from 343 to 127 ms but p95 remained 463 ms. The old
  shared-image route averaged 630 ms at full resolution, and flip-discard and
  waitable-swap-chain trials left the direct route's ~343 ms mean unchanged;
  none replaces it. Reflection keys now prove only their actual mirror pass,
  not an unrelated ordinary land pass: reflected work fell 117.7→106.0 ms
  mean across 114 calls, request mean 178→165 ms, and all seven saved images
  were byte-identical. Display mean remained 343→345 ms; this is a CPU-work
  improvement, not a demonstrated displayed-speed gain. With those narrower
  keys, a repeat 1.5 GiB reflection-cache trial
  cut reflected draws 89,293→49,741 and request mean 165→75 ms, but displayed
  mean only moved 345→322 ms and p95 worsened 979→1,262 ms; the larger cache
  was discarded. The narrower reflection key and zero-build prepared-world
  changes are now staged in the matching Renderer64 trio for a live game check.
  A one-pass full-viewport mirror failed on its second jump
  because the shadow receiver set exceeded the existing page budget; that
  trial was also discarded. An early GPU `Flush` after the camera copy moved
  much of the wait into the request (165→313 ms mean), while complete desktop
  latency worsened 345→361 ms mean across the same 100 destinations; it was
  discarded. These results identify a queue/surface handoff cost, not a reason
  to make the game thread wait earlier. Next, determine the
  driver/queue cost of binding the direct surface without sacrificing exact UI
  ordering or water quality; keep the WDDM budget trace for live calibration.
- The same live trace showed the inherited 1 GiB combined working-set limit
  assigning zero reusable unit GPU content on a full-screen x64 frame:
  attachments plus composition used about 1.10 GiB while the VM reported over
  7 GiB free physical memory. Renderer64 now allows a 1.75 GiB logical cache
  envelope, gated by physical-memory pressure, while Civ III/32-bit retains its
  old limit. At that captured occupancy the unit-content allowance changes
  from zero to 192 MiB; this is a budget calculation, not a measured FPS gain.
  Renderer64 also keeps its 75 MiB unit raster scratch across ordinary turns
  instead of reallocating it after one idle second. The direct visual path no
  longer copies the previous full-screen frame before an opaque full-screen
  replacement or Presents an unchanged retained frame. Retained native screen
  versions now skip identical full-screen re-commits in constant time, and a
  single full-screen source displays without an intermediate assembly texture.
  In a 640×480 GPU fixture, 80 unchanged transfers went from 80 compositions
  and 6.61 ms aggregate CPU submission to zero compositions and 0.05–0.06 ms;
  a changed transfer still updates exact pixels. The 2240×1260 single-source
  oracle now allocates zero assembly scratch. These are isolated path effects,
  not live FPS gains. The focused contracts,
  Windows compilation and staged startup probe pass; the matched trio is staged
  for install. Current live speed remains to be measured. The live trace
  separately shows native image leases disconnecting the retained animated map
  and resuming 180–230 ms game-thread redraws. The next trace distinguishes a
  DC lease from a pixel-pointer lease and times the necessary native readback;
  the ownership handoff is the next major idle-performance responsibility.
  The exact staged trio now has a refreshed short-capture readiness receipt:
  its controlled direct-surface input recording closed, exact and real-time
  same-build replays completed 7,137 calls and 87 accepted presentations,
  startup passed, and the read-only Windows launcher check passed. A separate
  full-screen, water-on fixture stopped at the known direct-surface versus
  legacy x86 desktop-pixel oracle before timed animation; it supplies no FPS
  comparison. Temporarily continuing past that oracle exposed a second fixture
  assumption: it expects the old GPU-owned JGL screen's CPU pixels to remain
  untouched, whereas the direct route did not satisfy that check. The bypass
  was removed; this fixture cannot qualify direct-route throughput or visual
  parity without a route-specific oracle. A same-input 384 MiB x64 unit-cache
  trial increased helper peak private bytes by about 88 MiB and worsened
  unit-call p95 (35.4 to 37.4 ms
  for unit input, 43.9 to 46.4 ms for body draws); it was discarded and was
  never staged. One short live run on this identified build must verify units
  and measure display/lease phases before choosing the next handoff change.
  The capture analyzer now reports those phases separately. On the older,
  incomplete 117-second capture it finds 82 map waits (101.7 ms median,
  221.1 ms p95), 82 native handoffs (200.1 ms median, 733.0 ms p95), and 16
  logged copy rejections, all recorded only as outstanding leases by that
  older build. The new staged build distinguishes DC from bits leases and
  times readback and direct-surface phases; the old trace cannot settle which
  ownership change is safe.
- A direct-surface timing audit exposed two avoidable idle limits: Windows
  condition-variable waits overshot the 16.7 ms target, and the retained map
  skipped visible-water samples inside inherited 15 Hz buckets. Renderer64 now
  uses an interruptible high-resolution waitable timer. Both production x64
  delivery and the shared-surface fallback now request the same 16.7 ms cadence
  with a short pause after slow frames; manual replay starts no autonomous timer.
  Visible water/shore motion samples at up to 60 Hz; authored resource poses retain their
  15 Hz frame selection. In the same controlled 2240×1260 scene with seven
  visible animations and no x86 visual requests, the staged predecessor logged
  44 map frames in three seconds (71.0 ms median spacing); the exact newly
  staged trio logged 128 (20.2 ms median, 43.2 ms p95). This is a renderer-side cadence
  comparison, not live displayed FPS or navigation latency. The previous trio
  is retained only as an ignored local rollback artifact. Ocean-wave integration
  passed 304 tests (two skipped), six production behavior cases, and a direct
  full-resolution lifecycle run. The new stage has a matching short-capture
  receipt, exact/real-time same-build replays and a passing launcher preflight.
  Next, measure the live DC/bits lease handoff and input-to-displayed-frame
  navigation before changing that ownership boundary.
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
  skipped), the approved injected compile passed after the startup correction,
  45 focused bridge/camera/cadence tests passed (one skipped), day/night unit behavior
  scenes passed, and an x86-to-Renderer64 direct-surface roundtrip accepted
  ordered birth, move, action/HP, retirement, stale-viewer rejection and ID
  reuse. An identical subsequent state caused no extra helper IPC call. No
  new CSV patch symbol was needed. Civ III gameplay was not launched by these
  checks, so first-move reveal timing, live surface/UI ordering and actual FPS
  remain unmeasured here.
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
