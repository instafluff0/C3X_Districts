# 64-bit renderer migration

**Decision:** use a separate x64 process as the target owner of renderer scene
state, preparation, assets and D3D rendering. Keep Civ III and its small C3X
capture/composition bridge x86. [Gate 1](helper64_gate1_results.md) proved the
Windows graphics/presentation boundary; [Gate 2](helper64_gate2_results.md)
proved real-core scene/pixel parity and a controlled address-space advantage.
Neither gate proved live gameplay FPS. The installed x86 path stays available
until the integrated x64 candidate passes.

## Target ownership

- **Civ III and x86 bridge:** authoritative gameplay/map state, native screen
  anchors, visibility, input/lifecycle hooks, native overlays/HUD, the game HWND,
  DirectComposition insertion, and a bounded fallback. Capture immutable value
  snapshots and changed records; never pass Civ III pointers or native object
  lifetimes across processes. Keep `injected_code.c` at hook, capture and
  insertion boundaries.
- **x64 renderer:** one retained world and dynamic scene, asset/geometry
  preparation, shared meshes/materials, animation clock, culling/pass selection,
  D3D device/resources, and final GPU map image. Visibility, unit/action,
  selection/path and camera inputs have explicit generation/lifecycle identities.
  Ambient water/resources continue while visible; unselected idle units remain
  frozen except work/action cases. Camera changes select prepared content instead
  of constructing it in the foreground.
- **Boundary:** versioned, bounded value messages with coalesced replaceable
  camera requests and durable world/gameplay changes. A small number of GPU
  shared textures carry completed map images; x86 imports/composes them with
  native UI. Backpressure never grows an unbounded queue or stalls Civ III's
  input thread. A helper crash/device loss invalidates its generations, retires
  handles and restarts or falls back without presenting a stale scene.

Users still launch the normal game executable. The x86 bridge starts the
packaged x64 helper and manages its lifetime; no second app or overlay window
is exposed to them. Both processes share the GPU and driver budget, so x64
address space is an enabler, not a substitute for render-path optimization.

## Migration sequence and acceptance

1. **Connect the complete replay.** Replace the scene-only trial driver with
   the existing x86 native composition owner feeding the x64 core, behind one
   game-window presenter. Run the same recorded native-interleaved workload on
   x86 control and x64 candidate, including camera, fog, units, selected/path
   overlays, autonomous animation, partial transfers and UI ordering. Validate
   image identity/pixels and input-to-first-correct-frame; trace x86, IPC, x64,
   GPU, composition and presentation time separately. No new gameplay claim
   rests on the isolated 51-scene timing.
2. **Move one scene owner at a time.** Transfer retained static world, dynamic
   objects, animation and their invalidation to x64 while preserving the native
   source-of-truth contracts. Prepare normal-world content at load/change time;
   camera requests perform selection/culling/submission. Measure readiness and
   bounded residency for Standard maps first, then report Huge separately.
3. **Reduce the measured whole-frame cost.** With shore waves and reflections
   on, tune pass selection, draw batching, shared instances, resource lifetime,
   GPU submission and native composition by the full replay's critical path.
   Use workers only where they shorten that path. Keep scene quality and
   authoritative unit/action timing. Track idle, scrolling and arbitrary jumps
   from any trigger, with the existing Standard-map <33 ms p95 coherent-frame
   goal and a separate stable-animation cadence check.
4. **Retire replaced machinery.** Inventory every x86 world compiler, viewport
   preparation/prefetch path, raster/pixel cache, CPU readback, duplicated scene
   owner and native saved-image copy. Delete a producer/storage path only after
   its x64 successor serves every caller, configuration-off behavior and reset
   case. Keep useful capture, replay, native composition and fallback contracts;
   retain any x86 image needed for native partial-transfer semantics. Record
   exact removals and surviving owners in the retirement ledger.
5. **Cut over the normal launch.** Check normal, x86-pressure, helper-crash,
   device-reset and config-off behavior in the full replay, then stage one
   identified candidate for a strategic live-game checkpoint. Compare x86 and
   x64/private and contiguous address space, GPU budget, first-use readiness,
   idle/scroll/jump p95 and visual correctness. Only then make x64 the default;
   keep a controlled fallback until the new path is stable. Do not treat the
   earlier x86 device-removal symptom as fixed without corresponding evidence.

Current integration checkpoint: the opt-in x86 bridge now consumes x64-owned
scene, camera, image, unit, tactical and visual results, including shared BGRA
presentation and the post-reset CPU/native route. A full native-interleaved
recording completed 15,503 calls, 1,007 accepted presentations and 3,549 clock
samples through the x64 primary path in strict forensic replay. Recorded clock
samples reach the helper; autonomous visual offers may produce an additional
valid frame relative to the x86 control. Of 309 compared CPU unit draws, 301
were pixel-exact and eight differed by one pixel each under the explicit pose
rounding audit. The x64 primary also completed the same full workload in
unpaced performance mode. These are integration and replay results, **not live
FPS or cutover acceptance**. The native screen handoff still suspends the
independent visual cadence, so seamless interturn animation is not yet proved.
The x86 and x64 presenters now support replay-only displayed-frame readback.
For the first 213 common presentation events, 209 whole-screen fingerprints
match; three initial exported BMPs are byte-identical. One of the four differing
events was inspected pixel by pixel and differs at exactly one channel of one
pixel. The other three have not yet been classified. Step 1 remains open for
the remaining displayed-image and presentation timing evidence;
steps 2–5 have not been accepted or cut over. The earlier
[shadow replay result](helper64_gate2_results.md#full-native-interleaved-shadow-checkpoint)
remains the independent scene comparison.

A focused 2240×1260 x86-window fixture held the owner thread without message
pumping for 450 ms after GPU presentation. The helper and x86 composition
endpoint completed 12 independent ambient frames, and replay-only readback
verified that the displayed map pixels changed. This proves the GPU-owned scene
can animate through a blocked window thread in that state; it does not prove
that Civ III's interturn/native-screen lifecycle preserves the same ownership.
The current native-screen handoff disables the cadence.

One paired unpaced full-recording run measured a 99.06 s x86-control envelope
and a 70.04 s x64-primary envelope. Sampled x86 private memory peaked at
2,456.5 MiB in control and 163.8 MiB with the helper; the helper peaked at
2,470.1 MiB. The x64 process completed 535 ambient offers versus 321 in the
x86 control; successful-offer p95 was 69.8 versus 127.8 ms. More completed
offers make the all-offer p95 incomparable as a frame-rate measure. The helper
spent 48.55 s in its own operations. Scene calls in the recording's
**post-reset CPU/native fallback** contributed 24.69 s; the 14 GPU scene calls
contributed 0.53 s. Ambient frames contributed 9.89 s and are the more relevant
repeated cost for the intended GPU path. This is one VM run per route, includes
replay reconstruction and different readiness outcomes, and does not establish
live FPS or a hardware GPU budget. It does show the intended address-space
separation and identifies visual-frame execution and native presentation for
step 3 measurement. The generated per-call evidence is under
`native/build/helper_trial/primary/`.

This migration takes priority over isolated x86 M3.8 tuning, but carries forward
M3.8–M3.13 stability, readiness, consolidation and navigation acceptance. It
does not begin deferred natural-wonder, constructed-wonder or District rendering.
