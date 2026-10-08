# Renderer64 performance review — October 7, 2026

The second smoothness pass, after a large round of visual work: rivers,
mountains, cities, units, farms, forests, roads and rails. The targets are as
close as possible to 60 fps for scroll, idle, zoom and map jumps, on both the
busy 1498 AD save and a light save. The earlier pass is
[performance review, October 4](performance_review_20261004.md).

## Method

- Busy save: `.cache/composition-integration-step/input-1498AD.SAV`. Light
  saves: `.cache/compiled-composition-step/light-initial-input.SAV` and the
  user's autosave `.cache/perf-review-20261004/user-3350BC.SAV`.
- Scripted `near` scenario (1× idle, 1× scroll on both axes, two minimap jumps,
  2× and 3× scrolls, 3× idle, zoom back, 1× idle) with `-MeasureCadence`, and
  with `-ProfileRenderer` for helper traces. Compare fps only at the same trace
  level.
- `python3 Renderer/tools/near_report.py CAPTURE` reports each segment: fps,
  adopted scroll steps (native and screen px/s, step interval, request to
  adoption) and minimap jump latency. It is tracked in the repository; the
  October 4 analysis scripts lived in `.cache` and were removed with it.

## 1. Vanilla scroll (user request)

Civ III's own edge scroll drives the camera again. Its timer fires every
66 ms (`on_timer_0x9F6500`), and Civ III's main update loop also calls it.
When the cursor is within 16 px of an edge, it moves the camera by a fixed
fraction of a tile, set by the scroll-speed option (¼, ½ or 1 tile), and twice
that at the very edge (`Main_Screen_Form::scroll_at_mouse`, `FUN_004de2a0`).

Removed:
- the custom 16 ms edge-scroll timer: elapsed-time speed, cursor-depth
  velocity, held motion clamped to 256×160 px (`C3X_NAV_PENDING` polling);
- image-space slides between steps (`PanTransition`, `set_pan`, the presented
  slide offset in the helper wire and in picking, `C3X_NATIVE_PAN_PRESENTED`).

Kept:
- deferred navigation. A scroll step is an asynchronous camera request, and
  Civ III's camera stays on the displayed view until the step is adopted, so
  labels, units and borders always register with the map;
- the scroll tag on the request (`C3X_NAV_REQUEST_SCROLL`). Until a step is
  adopted, each tick asks again for the displayed camera plus one step; the
  tag lets the in-flight step finish rather than being replaced.
- the 16 ms timer's zoom duties: the minimap box and the re-clamp after a
  zoom-out at an expanded map edge.

Consequence: Civ III advances one step per adopted camera job. Matching its
pace means adopting a step within one 66 ms tick. At 2× and 3×, Civ III's
tile-fraction steps are 2–3× larger on screen.

The helper wire version is now 16 (the slide field was removed).

**Tests.** `test_camera_navigation.py` checks that the scroll patch calls Civ
III's scroll with the request tagged, skips it during custom combat display,
and that the timer never moves the camera except to re-clamp after zooming
out. Removing the tag, the combat guard, or making the timer move the camera
each fail it.

## 2. Starting point (before this pass)

From the roads and farm sessions' `near` captures on the busy save (custom
scroll, trace level 2):

| Segment | October 6 (run133) | October 7 |
| --- | --- | --- |
| 1× idle | ~53 fps | 24–25 fps |
| 1× scroll | — | 14–17 fps; one 256 px step per 1.0–1.4 s |
| 3× idle | ~45 fps | 50–56 fps |
| Minimap jumps | ~1 s | 1.5–2.7 s |

**One 1× scroll step** (farm capture, QPC-stamped events only):
- the helper starts the camera job ~80 ms after the request;
- the job takes ~410 ms (32 tiles built, 1,048 reused);
- the native map pass at adoption takes ~217 ms;
- about one second from request to adoption.

**1× idle frames:** ~14 ms helper CPU render per frame (units 5.4 ms for ~97
units, reflection 2.6, water 2.1, preparation 1.9), yet only ~24 prepared
frames per second at the first idle location.

## 3. Vanilla-scroll baselines (October 7, build 7006f717)

`near`, `-MeasureCadence` (trace level 0) unless noted. Captures are in
`.cache/perf-review-20261007/`. Steps are adopted camera steps; "adopt" is from
a step's first request to its adoption.

| Segment | Busy 1498 (b01) | Light (l01) | User 3350 BC (u01) |
| --- | --- | --- | --- |
| 1× idle | 45 fps | 60 fps | 60 fps |
| 1× scroll x | 10 fps; 8 steps; 669 ms apart; adopt 504 ms | 49 fps; 70 steps; 79 ms; adopt 57 ms | 33 fps; 54 steps; 80 ms; adopt 58 ms |
| 2× scroll | 9 fps; 773 ms; adopt 605 ms | 52 fps; 79 ms | 34 fps; 79 ms |
| 3× scroll | 9.5 fps; 587 ms; adopt 452 ms | 47 fps; 80 ms | 35 fps; 79 ms |
| 3× idle | 51 fps | 60 fps | 60 fps |
| 1× idle end | 38 fps | 60 fps | 60 fps |
| Minimap jumps | 1.5 s / 1.1 s | (clicks miss on this map) | 38 / 24 ms |

On the light saves, vanilla's steps arrive every ~79 ms and are adopted in
~56 ms: the renderer keeps up with Civ III. The busy save is the problem.

**A busy step** (b02, trace level 2, p50): the camera request waits 162 ms in
the bridge's publication queue; the job takes 172 ms (native UI service turns
44, prepare 34, shadows 27, static 17, topology 11, mesh 7); the native map
pass takes 90 ms; first request to adoption is ~450 ms. Civ III's next tick
then comes 150–200 ms later because its thread is busy.

## 4. Findings in progress

- **Retained view during scroll jobs (hypothesis rejected).** During a
  camera job, ambient frames draw the retained completed view only when the
  job keeps the same camera origin. A scroll step never does, so its frames
  fall to the 125 ms in-job throttle (~8 fps) by design: the job shifts the
  static slots the old view would need. `retain_completed_scene` never
  refused (b04/b05: one 6.8 MB retention per run). The `completed-scene-refused`
  trace and the `C3X_RENDERER_RETAIN_VIEW_MIB` override remain for diagnosis.
- **Input-coverage backlog blocks the game thread.** During busy 1× scroll,
  native calls of ≥2 ms (tiny fills and lines that wait for the coverage
  worker's 512-operation backlog to halve) total ~107 ms per step: most of
  the native map pass. `Renderer/tools/hit_trace_report.py` summarizes a
  `C3X_RENDERER_HIT_TRACE=1` dump to find the worker's slow operations.
- **Camera-request overtake is limited.** A camera request passes queued UI
  image, tactical and present records only back to the first record it may
  not pass. Canvas-bound facts interleave with UI work, so the request still
  waits behind ~58 tactical and ~16 image records (~150 ms) per step.

## 5. Input coverage skips canvases Civ III never hit-tests

**Measured.** A `C3X_RENDERER_HIT_TRACE=1` busy `near` run (b07) applied 543k
coverage operations in 93 s of worker time over a 200 s run, with only 40
queries. The same trace replays in 9.8 s as an x86 build on the VM and in 2.6 s
on the Mac, so the in-game worker mostly waits for CPU while competing with the
helper. Two destinations took 88 of the 93 s:
- image 298, the `Units_Control` canvas: 200k unit-bar fills and 27k unit
  sprites (78 s);
- image 302, the screen canvas: the per-tick copy and keyed transfers from 298
  and from the main form canvas, plus fixed-UI transfers (10 s).
All 40 queries read image 235, the main screen form canvas.

**Why these two are never read.** Civ III's form hit test
(`get_form_under_mouse` → `FUN_00608d50`) reads only a form's own canvas
(`Base_Form + 0x274`), and only when Status1 lacks bit 2:
- the screen canvas is a standalone global (`p_jgl_screen_canvas`, 0xCAD030),
  not part of any form;
- `Units_Control` is created with flags 0x1000022 (`FUN_004e2b00`). Status1 is
  written only in the creation function, so bit 2 stays set and the canvas read
  is skipped.

**Change.**
- At the overlay's transfer onto the screen canvas, injected code declares both
  canvases with `C3X_NATIVE_HIT_EXEMPT` (137). The overlay is declared only
  while its Status1 bit 2 is set.
- The worker client skips later draws and uploads to those images before they
  reach the worker. The coverage model drops their history (`Scene::exempt`).
- If an exempt canvas ever becomes a source of another image, that image is
  exempted too, and the worker reports `native-hit-scene-refused`. A query there
  then fails closed (the form takes the input) instead of answering from
  incomplete history.

**Replay check.** The b07 trace replayed with 298 and 302 exempt from creation
takes 0.23 s instead of 2.64 s on the Mac, with all 40 recorded query answers
identical and no refusals.

**Tests.** `test_hit_exempt_canvases.py`: exempt canvases answer nothing and
ignore draws; other canvases are unaffected; reading from an exempt canvas
removes the reader's coverage; the worker client skips exempt draws in order;
the injected declaration keeps its screen, overlay and Status1 conditions.

**In game (b08 trace 0, b09 trace 2; same busy `near` run).** Civ III's thread
waited on the coverage worker 18 times for 0.27 s in total, against 619 times
and 9.6 s before (b02). Form hit queries now answer in under 2 ms instead of
20–50 ms, which helps hover and clicks. There were no refusals. Scroll pace did
not improve: steps stay 500–900 ms apart. The build also carries other sessions'
changes, so idle fps is not an A/B comparison.

## 6. Where a busy scroll step goes now (b09, b10)

One 1× step (b09, request to the next request, 657 ms):

| Stage | ms |
| --- | --- |
| Camera request waits in the publication queue | 145 |
| Helper camera job | 156 |
| Completion noticed by Civ III | 35 |
| Native map pass (Civ III's thread) | 153 |
| Civ III's next scroll tick | 161 |

On the light save the native map pass takes 2.7 ms (p50); on the busy save it
takes 104 ms (p50) and 253 ms (p90).

**Transport saturation.** During busy scroll the publication thread was busy
9.2 s out of 11 s (`publication-latency` service time):

| Record | Count | Total s | Mean ms |
| --- | --- | --- | --- |
| images | 533 | 3.8 | 7.2 |
| state | 753 | 1.7 | 2.3 |
| present | 96 | 1.7 | 17.3 |
| tactical | 1,619 | 1.5 | 0.9 |

- Each camera request waits behind 60–120 tactical and 17–31 image records.
  A canvas-bound unit observation (`unit()` with a target ticket) stops the
  request's overtake.
- Every native line becomes its own tactical record, and each one first
  flushes the pending image batch.
- All tactical records ran outside camera jobs. The transport is 82% busy even
  between jobs, so the cost is volume, not job checkpoints.
- The helper's own native-image execution is only 160–260 ms per 2 s. Most of
  an image record's 7 ms is round trips and waiting for the helper's worker.

**Bridge entries (b10, `native-entry-profile`, `entry_profile_report.py`).**
Bridge calls account for 14–18% of Civ III's thread time during scroll and
idle, so most of the native pass is Civ III and injected code. Within the
bridge:
- `C3X_NATIVE_VISUAL_POLICY` (op 116) averages 5–16 ms per call. That is
  unexplained so far, since the call only posts or reads a value.
- Native text (op 107) costs ~0.8 ms per call, about 100 calls per step.

The next capture splits op 116 (`visual-policy-wait`) and samples Civ III's
thread (`game_thread_sampler.h`, `C3X_RENDERER_SAMPLE_GAME=1`,
`game_samples_report.py`).

## 7. Civ III's thread is mostly inside the Windows compatibility shim

**Sampled profile (b11, `game_thread_sampler.h`, ~1 kHz, trace 2).** At busy
1× idle, 90% of Civ III's thread samples are inside `AcLayers.DLL`, the
Windows application-compatibility shim engine; during 1× scroll it is 75%.
Nearly all of them (45,000 of 46,700) sit at one offset in that DLL, which
looks like a loop inside one shim, not ordinary API work. The bridge itself
accounts for 3–8% of samples.

The stacks come from Civ III's per-unit animation tick inside
`Animator_update` (`FUN_004f08f0`, which walks `Animator.Units`):
- `FLC_Animation::tick` (`FUN_00402620`) re-initializes each unit's frame image
  every tick (`FLC_Frame_Image::FUN_005f7b60`). That clears it through
  `Sprite::destroy` (`FUN_005f7e80`, graphsy method 37), creates a new JGL
  sprite (`m32_create_sprite`), then decodes the frame. The destroy ends in the
  shim: 56% of idle samples.
- `Unit::tick_anim` (`FUN_005cbf50`, already wrapped by C3X) calls into the
  shim at `0x5cc422`: 32% of idle samples.

**The shim layers.** HKLM and HKCU both run
`Conquests\Civ3Conquests.exe` with
`DWM8And16BitMitigation DISABLETHEMES DISABLEDWM HIGHDPIAWARE WINXPSP2`, as set
by the GOG installation, so players' machines carry the same layers. In this
VM, Windows has also applied the Fault Tolerant Heap shim to the game after
earlier development crashes (performance review, October 4, environment notes).
That part, including the `HeapValidate` and `HeapFree` costs, is VM-specific,
so the measured cost is pessimistic for players. The per-tick sprite churn the
fix removes is real everywhere. Changing either shim setting is an environment
decision for the VM owner; this pass has not changed them.

**Next.** The sampler now also records the nearest calling module and the
import slot it calls through, so the next capture names the shimmed API.
Renderer64 draws unit bodies in 3D and suppresses the native body draw, so the
per-tick FLC frame rebuild produces pixels nobody shows. Candidate fixes:
- skip the native FLC frame rebuild while Renderer64 owns unit bodies. This
  needs a patch-table entry for `FLC_Animation::tick` (GOG `0x402620`), which
  the user would add;
- recycle JGL sprites of identical size through the Graphsy vtable (create
  `m32`, destroy `m37`). Runtime vtable hooks need no patch-table entry, but
  only help if the shimmed call is in create or destroy rather than in the
  sprite's own buffer setup.

**Shimmed call identified.** The JGL sprite destructor (`jgl.dll`,
`0x10007ed0`) calls `DeleteCriticalSection` through its import slot
(`0x100680b8`). Every JGL object owns a critical section;
`InitializeCriticalSection` and `DeleteCriticalSection` are called from 11 and
10 places in `jgl.dll`. Graphsy slot 32 creates a sprite (`operator new(0x50)`
plus constructor), and slots 34–37 all point to the same virtual delete.

**Fix: map unit frames rebuild only on animation change.**
`patch_FLC_Animation_tick_map_unit` replaces the call at `0x4F0AA2` in the
map animator's unit walk. Two patch-table rows were added with the user's
permission (`FLC_Animation_tick` define, `FLC_Animation_tick_map_unit` repl
call; see the [patch ledger](civ3_patch_dependency_ledger.md)). With custom
rendering on, the original runs only when the unit's frame holds a different
FLC than its current animation, or for the selected unit, whose frame the unit
panel shows. Tests: `test_unit_frame_rebuild.py` (config-off delegation,
unchanged frames skipped, animation change and selected unit rebuilt).

**First in-game run (b12) failed.** The tactical record joining (section 5)
counted each primitive as queue work. Behind a 2,776-packet backlog, route and
grid captures exhausted the publication's 65,536-unit work budget
(`async-publication-failed reason=renderer publication pressure`), and the GPU
image session stopped. Each tactical record counts one unit again; the 256 KB
join bound still caps a joined capture at about 4,000 primitives.
`test_large_tactical_captures_keep_one_work_unit_each` fails with
per-primitive units.

**Results (busy 1498, trace 0; b08 before, b13 after).**

| Segment | b08 | b13 |
| --- | --- | --- |
| 1× scroll x | 6 steps, 797 ms apart | 14 steps, 354 ms apart |
| 1× scroll y | 8 steps, 547 ms; 9 fps | 15 steps, 228 ms; 28 fps |
| 2× scroll | 4 steps, 1,229 ms | 8 steps, 609 ms |
| 3× scroll | 5 steps, 1,037 ms | 10 steps, 535 ms |
| 1× / 3× idle | 41 / 44 fps | 42 / 49 fps |
| Minimap jumps | 1.3 / 1.1 s | 1.7 / 1.7 s |

Civ III's CPU fell from about one full core (p50 97%) to p50 23% (b12
`typeperf`). A busy step now breaks down (b14, trace 2, p50) into: queued
90 ms, job 96 ms, native pass 101 ms, request to adoption 260 ms (from 162,
172, 90 and 451). The light save is unchanged or slightly better (l02:
1× scroll 53 fps, 3× scroll 51 fps, idle 60). Jumps on the busy save are slower
in this build, which also carries other sessions' new content. That is still
being checked.

**Where Civ III's thread goes now (b14).** In 1× scroll about half its samples
wait in its message loop. AcLayers fell to 15%: the remaining sprite
destroys (selected unit, one-shot animations), `jgl.dll`'s `HeapValidate`
calls, and calls from the bridge. During native map passes two-thirds of the
samples are inside system calls and 22% in the bridge. Native text (op 107,
~1 ms a call, ~67 calls a second while scrolling) is the costliest bridge
entry.

**Second unit walk call and drawn-FLC rule.** The b14 samples showed two
remaining rebuild paths: the walk's second call (`0x4F0AF0`, now patched too),
and units whose current animation has no loaded FLC data. For those Civ III
draws `Animations[1]` instead, so comparing with the current animation never
matched. The patch now compares with the FLC the original would draw.

## 8. Native text cache (32 entries thrashed)

The sampler's module chains (v3, with the bridge's linker map) showed native
text time inside `c3x_native_text::compile`. Every draw was rebuilding its
raster: 17 GDI renders, two GPU images and two uploads per label. The adapter
cached only 32 strings, while a busy map draws a few hundred distinct strings
(city names, sizes and production, each with colour variants). The cache now
holds 512 entries within 16 MB; a label raster is about 20 KB.
`test_busy_map_label_set_stays_cached` fails with 32 entries.

Native text dropped from 1.66 s per 1× scroll window (1,567 calls, ~1 ms each)
to below the four costliest bridge entries (b15 → b17). The busy native map
pass (passes over 10 ms) fell with it:

| Build | p50 | p90 |
| --- | --- | --- |
| b08 (hit exemption) | 104 ms | 253 ms |
| b13 (unit frame fix) | 81 ms | 158 ms |
| b16 / b18 (text cache) | 18–19 ms | 89–154 ms | Transport busy time
during 1× scroll fell from 87% (b14) to 72% (b17) of the window.

## 9. Where things stand (October 8, early morning)

Busy 1498 AD save, trace 0:

| Segment | b01 (start) | b16 (now) |
| --- | --- | --- |
| 1× idle | 45 fps | 42 fps |
| 1× scroll x | 8 steps, 669 ms apart | 10–14 steps, 354–601 ms apart |
| 1× scroll y | — | 15 steps, 228–236 ms apart; 20–28 fps |
| 2× scroll | 773 ms | 470–609 ms |
| 3× scroll | 587 ms | 316–535 ms |
| 3× idle | 51 fps | 49–53 fps |
| Minimap jumps | 1.5 / 1.1 s | 1.4–2.1 s |

The build carries other sessions' new content (volcano smoke, combat effects,
ground states, tunnels removed), so idle and jump rows are not pure A/B.
Civ III's thread is no longer the limit: in busy 1× scroll it waits in its
message loop about half the time. What limits now:
- **Transport** (still 72% busy during busy scroll): native UI image batches
  (~7 ms each), synchronous camera-begin (~28 ms) and camera-adopt (~37 ms).
- **Camera job** (~100 ms per step): shadows, preparation, statics.
- **Busy idle frames** (~20 ms each): about 10 ms of CPU and 6–13 ms waiting
  for the GPU fence. The fence wait is not GPU throughput (see below).
- **Far jumps:** the native pass waits for a 500–1,000-tile camera job
  (~1.4 s).

Light saves keep pace with vanilla: steps 79 ms apart at 50–53 fps scroll, and
60 fps idle.

**GPU timeline at busy idle (b20, `C3X_RENDERER_PROFILE=1`).** The readback
timeline serializes CPU and GPU, so only its phase times count, not its frame
rate.

| Segment | GPU work per frame | of which scene preparation | Collection wait |
| --- | --- | --- | --- |
| 1× idle | 3.5–4.6 ms | 2.9–3.9 ms | 30–35 ms |
| Scroll and zoom | 3.2–6.6 ms | up to 8 ms | 20–86 ms |
| Zoomed out, static | 0.7 ms | 0.1 ms | 31–34 ms |

Scene preparation (shadow, instance and unit preparation before the first
draw) is about 80% of the GPU work, and every pass after it is under 0.2 ms.
The wait for a readback stays near 32 ms whether a frame holds 0.7 ms or
4.6 ms of GPU work: about two 60 Hz host intervals. Parallels completes
readbacks at its presentation cadence. So the busy idle fence wait is
translation-layer pacing, not GPU throughput. Busy idle gains must come from
the helper's ~10 ms CPU frame. A frame that misses a host interval waits for
the next one; a mix of 60 and 30 fps frames would average the measured 42 fps (inferred, not traced per frame).

## 10. Unit movement at native speed (user request, October 8)

Goal: from releasing the input to the unit arriving, moves, reveals and combat
take exactly as long as in native Civ III, keeping the renderer's turning and
easing.

**What was slower.** A step's clock started at its first displayed sample, so
transport and camera delays came first. A 60–180 ms idle turn then preceded
travel, and holds during camera preparation added their whole length. Each
queued step started after the previous visual step, so lag accumulated along a
path.

**Changes** (`render_core/unit_instances.h`, `unit_locomotion.h`; contract in
[scene and motion](renderer64_scene_and_motion.md)):
- **Start.** A step starts at its move event's QPC time. The worker reports the
  offset between QPC and the scene clock, live only, never during replay. A late
  first sample skips at most a quarter of the travel.
- **Turn.** The facing turn runs during travel.
- **Native overhead.** Civ III confirms a step after its animator snaps the
  unit onto the tile, 90–160 ms after start plus travel in the VM. The smoothed
  measured overhead stretches later steps' easing, so they arrive when Civ III
  confirms them, and a path never pauses between tiles.
- **Holds.** After a frozen view, a step runs at double speed until the held
  time is recovered.
- **Diagnostics.** `unit-arrival` and `reveal-shown` helper traces (trace
  level 1 or higher).

**Tests.** `test_unit_motion.py` `test_travel_keeps_native_start_and_duration`
covers native start, the quarter-skip bound, queued steps, hold recovery, the
learned overhead and replay fallback. Each part was mutation-checked: removing
anchoring, catch-up or the stretch fails it. The late-turn expectations changed
from an idle hold to immediate travel. `test_unit_arrival_visibility.py` covers
the reveal count.

**In game (`unit-turn`, user 3350 BC save, trace 2).**

| Run | Step | start_lag | travel | arrival vs Civ III's confirmation |
| --- | --- | --- | --- | --- |
| m01 (native start, no turn hold) | 1 / 2 | 0 / 0 ms | 569 / 569 ms | −156 / −113 ms |
| m03 (plus learned overhead) | 1 / 2 | 0 / 0 ms | 569 / 729 ms | −160 / +27 ms |

The first step of a session arrives up to the overhead early, never late;
later steps land within one animator tick of Civ III's confirmation.

**Reveals (still slower than native).** Newly explored tiles appear 98–362 ms
after Civ III's visibility capture (m04). Native redraws them on its next
update. A one-cell reveal ran a camera job that rebuilt 31 tiles (21 without
geometry, 10 neighbours whose appearance changed): cliffs 81 ms, terrain
preparation 45 ms, features 22 ms, then waves and shadows. The world change
reached the helper 66 ms after the capture. Hidden tiles keep masked topology
(`ViewerTopology`) and no geometry, so a reveal must build them. Section 11
explains why preparing them during travel does not pay off, and what did.

**Not measured.** Input-to-move latency (key or click to Civ III's move event):
the scripted harness does not timestamp posted keys. Combat clips already follow
Civ III's own cycle durations and waits, and the half-tile approach uses native
travel speed. Combat was not re-measured in this pass.

## 11. Second pass (October 8, morning): items 1–5

The user asked for (1) faster reveals, (2) input and combat timing checks,
(3) faster scroll steps, (4) busy idle, and (5) faster far jumps. Windows
compatibility settings stay as they are.

**Input-to-move latency (item 2).** The harness now records posted keys with
QPC stamps (`key-events.json`), and `unit-arrival` traces carry
`shown_lag_ms` (move event to the first scene sample holding the step) and
`event_qpc`. `Renderer/tools/unit_timing_report.py` pairs them. On the user
save (m05, m06): key to Civ III's move event 0.5–2.4 ms, move event to first
sample 2–15 ms. Input latency is not a problem.

**Frames in flight (items 3 and 4).** Busy idle frames (b19, trace 2) spent
about 8 ms preparing and 2 ms evaluating the scene, then on about half the
frames waited 11–23 ms to bind the swap-chain buffer and up to 17 ms at the
GPU fence. Both waits release on Parallels' 60 Hz presentation ticks, about two
ticks after submission, so with two frames in flight a slightly long frame
missed a tick. Three frames in flight (four buffers, latency 3, three-entry
fence ring; `trial_frames_in_flight`):

| Busy segment (trace 0) | 2 frames (b18, b21, b23) | 3 frames (b22, b24, b25) |
| --- | --- | --- |
| 1× scroll x step | 407 ms | 334 ms |
| 1× scroll y step | 283 ms | 182 ms |
| 2× scroll step | 618 ms | 534 ms |
| 3× scroll step | 543 ms | 438 ms |
| 3× idle | 45 fps | 52 fps |
| 1× idle | 24–42 fps | 39–42 fps |

The light save stayed at 60 fps idle and vanilla step pace, with 2× and 3×
scroll at 57 and 54 fps (was 52 and 51; l03). The user save stayed at 79 ms
steps, 40–44 fps scroll and 60 fps idle (was 36–39 fps scroll; u03). The cost
is one more frame of display latency. Test:
`test_native_visual_cadence.py` `test_three_frames_in_flight_share_one_depth`.

**River pages (item 1).** Every reveal changes the masked topology revision,
and `NaturalWorld::update_rivers` dropped all river pages (16 in the main
world, 2 per query scratch), so the first queries after a reveal rebuilt every
page. Each page records the exact cells and river flow it read, so a page whose
inputs are unchanged would rebuild identically. Pages are now kept unless one of
their inputs changed (`lab/shared/natural/world.h`; tests
`lab/shared/natural/test_world.cpp` and `test_ground_preparation.py`).

**Reveal timing after both changes (m06).** First reveal 291 ms after Civ III's
visibility capture (was 362–381), second 56 ms (was 98–104). The first reveal's
job took 281 ms, but 189 ms of it was display frames serviced inside the job
(command 12, about 12 scene frames); its own work was about 92 ms.

**In-job frame cap.** Camera jobs service display frames on their own thread.
With a retained view they ran at full rate; now they are capped at 30 Hz (the
8 Hz cap without a retained view and the zoom exemption are unchanged).

| In-job frames | Reveal job | Frames inside it | First reveal shown |
| --- | --- | --- | --- |
| uncapped (m06) | 281 ms | 189 ms | 291 ms |
| 30 Hz (m07) | 161 ms | 76 ms | 187 ms |
| 20 Hz (m08) | 136 ms | 61 ms | 169 ms |

30 Hz was kept: 20 Hz gained only 18 ms for visibly choppier animation during
jobs. Confirmation with the final build (m10): first reveal 192 ms, second
57 ms, second step arriving 8 ms before Civ III's confirmation. On the busy save (b29) scroll steps stayed within the three-frame range
(304, 231, 471 and 476 ms at 1× x, 1× y, 2× and 3×), and jumps took 0.91 and
2.17 s. Test: `test_native_visual_cadence.py`
`test_direct_visual_busy_is_distinct_from_unchanged`.

**Busy 1× idle with three frames in flight (b28, trace 2).** Frame interval
p50 19.4 ms (was 21.7), fence wait p50 0.9 ms (was 6–11). Each frame still
spends about 10 ms of CPU (scene 7.0 ms: units 2.8, water 1.4, reflection 1.3,
preparation 1.1; evaluation 2.0), and about a quarter of frames wait 9–16 ms
to bind a back buffer.

**Reveal preparation during travel (not pursued).** Building hidden cells
early does not survive the reveal: the revealed record adds era, resources,
roads and territory, so tile signatures and object keys change, and
`CapturedScene::publish` clears compiled records. Only neighbour meshes and
the ground component could be reused, and vegetation neighbours depend on the
revealed cells' appearance revision anyway.

**Far jumps (item 5).** A busy jump job (b19) handled 1,080 tiles for a
viewport of about 330: it restored 1,172 prepared tiles from the compressed RAM
store (1.2 s of worker time on four workers), built 673 GPU meshes (1.45 GB,
upload 281 ms), built shadows (130 ms), then a 398 ms display frame inside the
job waited for the job's own GPU uploads. Moving off-screen halo tiles to the
background worker (`C3X_RENDERER_PREFETCH_FOREGROUND_CONTROL=2`) was
inconclusive in one run (jumps 0.96 and 3.5 s against 0.94–1.36 and 2.2–2.5 s).
The `near` jump figures on the light and user saves (4–10 s, or 24 ms) are
measurement artifacts of that scenario on those maps.

With today's defaults (b28, trace 2) the two busy jumps took 772 and 1,437 ms:
restoring prepared tiles 0.9–1.3 s of worker time, GPU uploads 120–200 ms
(about 1 GB each), shadow page proofs 84–206 ms, features and terrain
preparation about 130–160 ms, and 117–310 ms of display frames inside the job.
The zoom-out job took 2.5 s, 2.1 s of it in 153 display frames: a zoom keeps
every frame, which starves the job (zoom out to 1× runs at 16–23 fps). After
each zoom step the frames also rebuild the static composition at the new zoom
(spread over many frames, 4–10 ms each) and the reflection (up to 12.5 ms), so
frames there cost 8–23 ms instead of about 7 (b30).

**Combat (item 2).** Civ III's combat waits are clock-paced:
`FUN_004f0010` calls the animator update in a loop until a QPC deadline in
milliseconds passes, so extra CPU per update means fewer updates, not a longer
combat. A melee run on the user save (c01, game-thread samples) took 8.0 s from
Civ III's combat start to end. Before the attack the game thread idled in its
message loop (89%); during it, 58% of samples were the selected attacker's
per-tick sprite rebuild (`FLC_Animation_tick` → `FUN_005f7b60` → jgl.dll →
the compatibility shim's `HeapValidate`), about 6 ms per update. That is well
under an FLC frame interval (50–66 ms), so animation pacing and combat length
match native. The 3D clips follow Civ III's sampled animation state.
`game_samples_report.py` now segments around the scripted attack key.

**Busy scroll steps: the native image queue (b31, trace 2).** Step p50: wait 16,
queued 38, send 11, job 75, native pass 30, request to adoption 179 ms. In the
2× segment camera-begin records waited 225–355 ms. The chain: a camera-adopt
must follow the native image batches drawn for the ticket it retires, and a
camera-begin may not pass a pending adopt. Image batches back up during scroll
(queue p50 233 ms at 1×, 414 ms at 2×, 241 ms at 3×) because most execute in
under a millisecond but some take 200–386 ms (helper execution p90 31 ms,
p99 96 ms). They are not large uploads (native source refreshes during scroll
move about 0.1 Mpx per 2 s); a batch waits inside the helper for a camera job
checkpoint. `camera-service-turn` now reports `wait_ms` (submission to the
checkpoint that served it) to find job phases that run long without one.

With `wait_ms` (b32): image batches (command 9) served inside camera jobs wait
little at checkpoints (p50 0.1 ms, p90 8.5 ms, 10 of 374 over 50 ms). The slow
batches come from GPU drains. A display frame's fence check blocks until the
GPU finishes all queued work (Parallels ignores `DO_NOT_WAIT`), so a frame
right after a heavy job submission (static composition, uploads) held the
renderer gates for 170–186 ms. An image batch that touched the GPU inside a job
ran 268 ms for the same reason. Parallels runs GPU work in order, so these
waits only move unless the job's GPU work shrinks. They are outliers: a typical
busy step is the job (75 ms), Civ III's next native pass (30 ms) and ordinary
queueing (38 ms).

## 12. Light-save scroll: every step on Civ III's next tick (October 8)

User priority: light-save scroll as close to native Civ III as possible, for a
new-game demo.

**How native scroll is timed.** Civ III's 66 ms edge-scroll timer is a USER
timer (`FUN_006205d0` uses `timeSetEvent` only below 50 ms), so it fires on
the 15.6 ms system tick: every 78 ms. Each tick runs the animator update
(`FUN_004eeb20`), then the scroll handler (`FUN_0046bf10` → `FUN_004de430`).
Native `move_camera` (`FUN_004df700`) only stores the camera and sets a dirty
flag, so native Civ III also draws a step at the next tick's animator update.
Our poll for a finished step runs in that same update
(`patch_Animator_update_display`); a step that is ready by then is drawn when
native would draw it (about 57–61 ms after the request).

**What was slower than native.** 20% of light-save steps were not ready by the
next tick and slipped one more (138–144 ms). Each slip also lost a step:
Civ III re-requests the undrawn camera on that tick. Three of four slips sat
behind a display frame serviced inside the step's camera job, which waited
18–28 ms on Parallels GPU pacing (fence or back-buffer bind); the fourth
request arrived only 25 ms before the tick.

**Change.** A camera job's start counts as its last in-job frame for the 30 Hz
cap, so jobs shorter than 33 ms (light-save steps take 10–20 ms) run without a
display frame inside them. Test: `test_native_visual_cadence.py`
`test_camera_job_start_counts_as_an_in_job_frame`.

| Light save (l14, l15) | Before | After |
| --- | --- | --- |
| Steps drawn later than the next tick | 20% | 1–2% |
| 1× scroll | 2,525 px/s | 2,782–2,803 px/s |
| 2× scroll | 2,818 px/s | 3,009–3,020 px/s |
| 3× scroll | 1,538–1,663 px/s | 1,798–1,802 px/s |
| Scroll fps | 52–57 | 53–57 |

The new-game save (n02) went from 3,220 / 2,918 / 1,687 px/s to 3,362 / 3,009 /
1,782 px/s at 1× / 2× / 3× (3% of steps late; 1× still 10%). On the busy save
(b33) steps were as fast or faster (2× 464 ms, 3× 316 ms) but 1× y scroll
fell to 22.5 fps (about 30 before): a 75–100 ms busy step job now has one
fewer display frame inside it.

**Remaining difference from native (l17, route witness).** After Civ III's
tick draws a step, the frame showing the new map is presented 38–48 ms later
(frames at +6 and +14 ms still show the old map, then no presentation for
about 34 ms). Native shows its step within the same tick.

**Tried and rejected.** Polling from the renderer's view timer between ticks
(made adoption no earlier: `Animator_update` returns early outside Civ III's
own schedule, and the timer's extra work slowed steps to 92 ms). Raising the
process timer resolution so the scroll timer fires at 66 ms would scroll
faster than native, not respond faster.

Note: `injected_code.c` changes reach scripted game tests only after
`INSTALL.bat`; `TEST_INJECTED_CODE_COMPILE.bat` only compiles.

**Busy-save scroll quality (b34, 10 Hz window frames; deferred by the user).**
At 2× zoom, frames during scroll are visibly softer than the settled view:
railway ties, field textures and buildings blur and city smoke is missing,
while native text stays crisp. This looks like the static layer's resampled
stand-in showing while full-quality regions are rebuilt. 1× and 3× frames were
sharp. Separately, the first step of a 1× or 3× scroll took about a second to
appear (identical frames for about a second after the cursor reached the edge).

**Light-save scroll after the front bypass (l19, l20).** A newly committed
native front (an adopted step and its overlays) now bypasses the in-job frame
hold. Civ III's draw to the new map's presentation: p10 8.5 ms, p50 44 ms,
p90 64 ms; late steps 2%. The user judged light-save scroll good enough for now
and moved on to zoom.

## 13. Zoom (October 8)

`Renderer/tools/zoom_report.py` reports per wheel notch: start (wheel to the
first moved presented zoom), settle (within 1% of the final zoom), fps and the
longest frame gap during the transition, and fps in the second after it.

Light save baseline (l20):

| Notch | Start | Settle | Transition fps | Longest gap | After fps |
| --- | --- | --- | --- | --- | --- |
| In 1→1.25→1.5→1.75→2 | 29–31 ms | 171–187 ms | 28.5–42.4 | 46–93 ms | 48–56 |
| In 2→3 | 30 ms | 187 ms | 31.9 | 32 ms | 54 |
| Out 3→2.5→…→1 | 28–45 ms | 140–216 ms | 26.7–42.2 | 32–63 ms | 47–51 |

Response and the 166 ms ease are as designed; frames during the transition are
the problem.

Light-save transition frames (l21, trace 2) cost only 2–3 ms of scene CPU,
but display frames took 19 ms (fence wait 8.6 ms, frame 8.8 ms) against 6.3 ms
at idle (2.8 and 3.4). The static layer promoted a freshly refined raster on
almost every transition frame. Capping refinement with
`C3X_RENDERER_REFINE_PIXELS` (400,000 and 150,000 pixels per frame, l22/l23)
did not raise transition fps (27–47), so refinement is not the limit. Shadows
are not rebuilt during transitions. `C3X_RENDERER_PROFILE=2` now reports the
GPU timeline for every frame, so short transitions are not averaged away.

**Per-frame GPU timeline (l24, `C3X_RENDERER_PROFILE=2`).** GPU work per frame
is nearly the same in zoom transitions (3.43 ms; scene preparation 2.77) as at
idle (3.19; 2.59). Zoom display frames (l21) instead wait on presentation:
back-buffer bind 9–44 ms on some frames (0.05 ms at idle) and the GPU fence
11–22 ms on others, while CPU work stays at about 5–9 ms. Hypothesis (not yet
proven): Parallels' presentation cost grows with the share of the frame that
changes. Zoom changes every pixel every frame; light idle changes little and
holds 60 fps. Busy idle (wide animated water), scroll-step frames and the
44 ms from Civ III's draw to a presented step fit the same pattern.

**Hypothesis rejected (l25).** With every presented pixel changed on every frame
(the display shader flipped the lowest blue bit on alternate frames; an
experiment since removed), the light save still idled at 60 fps at 1× and 3×,
and zoom transitions stayed at 25–48 fps. Parallels' presentation cost does not
depend on how much of the frame changes. The zoom-frame waits come from work
specific to zoom frames.

## 14. What a frame costs in the VM (October 8)

**In-order readbacks cannot split a frame in Parallels.** The l24 timeline
(3.4 ms per zoom frame) mapped its marks only after the whole frame was
submitted. Parallels appears to finish a batched frame at once, so those marks
mostly measured the tail. `C3X_RENDERER_PROFILE=3` now waits for the GPU at
each mark instead and reports each phase's total and its mark count. The waits
serialize CPU and GPU and add about 0.25 ms per mark, so the numbers are each
phase's isolated cost, not a frame time. `C3X_RENDERER_DRAIN_PROBE=1` does the
same after every helper command.

**Cost model (`tools/d3d_pass_cost_probe.cpp`, run in the VM).** Medians per
frame of work, each followed by a completion wait:

| Work | 25 items | 100 items |
| --- | --- | --- |
| 64×64 quads on one full-screen target | 0.47 ms | 1.4–3.1 ms |
| 64×64 quads alternating two full-screen targets | 1.6 ms | 8.6–12.9 ms |
| 64×64 quads alternating two 256×256 targets | 1.6–2.4 ms | 11.5–12.5 ms |
| 64×64 copy, then a draw, repeated | 3.8–4.9 ms | 16.9–17.4 ms |
| compute dispatches over 64×64 | 0.6–1.0 ms | 0.7–2.7 ms |
| dispatch with a 64×64 copy before each | 1.5 ms | 6.1 ms |
| dispatch, then a draw, repeated | 3.7 ms | 17.3 ms |

Full-screen copies and draws cost 0.15–0.25 ms each. In Parallels, every
change of render target, and every switch between drawing and copy or compute
work, costs about 0.1–0.17 ms whatever the target's size. Copies and dispatches
run back to back cost 0.01–0.06 ms each. Pass structure matters; pixel count
barely does, which also explains l25.

**Light save, isolated costs (l29, l30).**

| Phase | Idle | Zoom frame |
| --- | --- | --- |
| Scene (preparation to reconstruction) | none | about 5.4 ms |
| Composition prepare | 0.3 ms | 1.9 ms |
| Composition evaluate | 5.7 ms | 9.3 ms |
| Composition assemble | 2.8 ms | 3.8 ms |
| Display | 1.2 ms | 1.0 ms |
| Image batch (each) | 3.7 ms | 3.4 ms, p90 9.4 ms |

Idle frames re-evaluate about 21 native UI operations every frame (panels,
buttons and text that blend over the animated map, over a full-screen
quantized copy of the map), with about 200 copies and 14 million copied pixels
per frame. During zoom, Civ III also redraws (the minimap viewport changes
with every tick), so about three image batches arrive per frame. Together that
is roughly 30 ms of GPU and translation work per zoom frame against about 10 ms
at idle, which matches the 27–47 fps transitions.

**Tried and rejected: compute copies in composition (October 8).** In the
probe, an interpreter-style operation (a 64×64 snapshot copy, a cleared
scratch and updated constants before each dispatch) cost about 0.13–0.22 ms,
but 0.04 ms with compute copies and mapped constants. Moving native-operation
snapshots, scratch clears, constants and retained assembly copies to compute
was bit-identical (VM exactness test), but in the game it did not help.
Back-to-back runs on one build, switching with an environment variable:

| Light save | Copy engine (l38, l41) | Compute (l33, l40, l42) |
| --- | --- | --- |
| 1× scroll x | 53.8–55.8 fps | 44.6–47.1 fps |
| 2× scroll | 52.4–54.9 fps | 46.7–49.2 fps |
| 3× scroll | 52.9–53.9 fps | 46.2–48.0 fps |
| Zoom transitions | 24–50 fps | 19–52 fps |

The busy save showed no difference (b36, b37). The change was reverted.
Isolated costs, whether from the probe or `C3X_RENDERER_PROFILE=3`, did not
predict whole-frame time in Parallels: part of the cost is CPU time in the
translation layer at submission. Only back-to-back game runs decide.

**Fence waits mean the GPU is behind.** Mapping an older, completed fence
returns in about 0.6 ms even with 100 render-target switches queued after it.
So the 10–14 ms gate waits in zoom frames mean the frame from three frames back
has really not finished: the VM's GPU and translation work is behind.

**Scene-hold diagnostic (l43, l44; inconclusive).** A diagnostic switch
(`C3X_RENDERER_DIAG_ZOOM_SCENE_HOLD=1`, wrong picture) was meant to
re-render the scene at most every 120 ms during zoom. Transitions stayed at
21–48 fps (control 19–48). However, unit screen poses change with zoom, and a
pose change vetoed the hold, so it probably never applied. Rerun without that
veto (l47, l48): transitions 25–40 fps with the hold, 21–42 fps without;
scroll, idle and jumps unchanged. Holding the scene does not raise zoom fps,
so the limit is elsewhere in each zoom frame (composition, presentation or
native traffic). These runs carry no per-frame trace to show how often the
hold applied.

**Jump timing fix.** Civ III can move the camera on the minimap button press,
before the release. `near_report.py` timed jumps from the release, so it
skipped those moves and reported the next distant handoff as a 4–10 s jump
(l38–l44). It now times from the press and reports no jump when Civ III
ignored the click (test: `tools/test_near_report.py`). Light-save jumps take
37–80 ms.

**What heavy composition frames execute (l45, `C3X_RENDERER_TRACE_EVALUATED=1`).**
The 110–120-operation frames follow every Civ III redraw, during scroll as
much as during zoom. They re-execute Civ III's bottom interface: per frame,
16 fills, 16 copies, 8 blends and 8 images of 32×32 (the command buttons),
four 36×30 groups, five text labels and the large panels (510×236, 470×224,
440×64, 300×300, 294×286). Recipe reuse matched only 425 of 5,353 probed
recipes (8%), so each native redraw rebuilds the interface even when it looks
the same. These frames are not specific to zoom.

## 15. Busy-save camera steps before camera decoupling (b46, October 8)

Trace level 2 with route witness, `near` scenario on the 1498 AD save.
`step_report.py` medians per scroll-step job: request to job start 28.6 ms,
camera-begin record queued behind other publications 121 ms, its delivery
21 ms, the job 94 ms (shadow 16, scene preparation 22, static 12, mesh 6),
then Civ III's next native map pass 37 ms. Request to adoption: 190 ms p50,
115–900 ms overall. Scroll: 1× x 193 px/s at 8.5 fps, 1× y 382 px/s at
15 fps, 2× 231 px/s, 3× 168 px/s; idle 32–44 fps; jumps 1.6 and 2.4 s.
Most of each step is waiting in queues and on the job, which camera
decoupling (G1) removes from the camera's path.

## 16. Camera decoupling, stage 1: camera requests during image waits (October 8)

**Change.** The bridge's single transport thread sends every publication to
the helper synchronously. A camera request may run ahead of queued UI work,
but it still waited for the entry in flight. While an image batch executes in
the helper, the transport thread now sends a camera request posted meanwhile
(`Publication::run_passing`, woken by `SceneClient::interrupt_image_wait`),
and the helper accepts a camera begin while a batch is outstanding.
`C3X_RENDERER_CAMERA_IMAGE_WAIT=0` turns it off for A/B runs. Tests:
`test_async_publication.py` (`run_passing` order, the helper rule).

The first build broke the busy save: the helper's reliable-prefix rule
rejected the camera request ("reliable prefix requires image execution
receipt") and the publication faulted, so no map was drawn (b50, b52). Every
capture is now checked for adoptions, zoom targets and publication or
operation failures before its numbers are used.

**Result (busy save).** Camera-request queue wait p50 121 → 70 ms, p90 169 ms
(b46 → b58, both trace level 2). The rest of the wait is the previous step's
synchronous adoption call (38%), image batches the lane could not pass (40%)
and synchronous presents (21%). Request to adoption p50 190 → 175 ms. Steps
still land every 3–4 ticks (232–312 ms, b54, b55), so scroll speed is
unchanged.

**What gates a busy step.** A step is adopted on the first tick after its
camera job finishes. The job (82 ms p50) is mostly per-camera preparation,
not drawing: on a frame at a new camera, scene preparation takes 20.7 ms
(1.0 ms at an unchanged camera), static strips 11 ms (0.06), reflections
5.3 ms (1.2). The shadow-page build for the new strip is 14 ms p50 (35 ms p90:
refresh 2.4, proofs 5.5, casters 2.1, draw 3.1; 8 pages, 2,640 draws).
Next: prepare the next step's strip ahead of the request (no visible change),
then decide on stand-ins.

**Separate issue.** Since about 13:50, scripted wheel input on the busy save
registers no zoom target (no `zoom-target` trace), with this build and
without it (b53, base build). Scroll and jumps are unaffected. Not yet
explained.

**In-job frame cap (b59–b62).** Lowering the retained-view in-job frame cap
from 30 to 8 Hz (`C3X_RENDERER_JOB_FRAME_HZ`, for A/B only) did not speed up
busy steps: 1× x 391–868 ms at 8 Hz against 316–407 ms at 30 Hz, other
segments equal within noise. The default stays at 30 Hz.

**Static layer on camera steps (b58).** On busy step frames the static
layer's 11 ms is mostly recentering, not missing strips: only 14 of 72 step
frames had missing area, but 39 recentered (copying the raster to a slot
around the new camera and recomputing its dependencies) and only 33 fit the
existing slot. The slot margin is 320 × 192 px, so a 128 px step overruns it
every two or three steps. A scrolling (wrap-around) raster would remove
recentering.
