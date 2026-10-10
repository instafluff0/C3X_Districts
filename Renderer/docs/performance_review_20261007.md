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

## 17. Camera-step glide restored behind a flag (October 8)

At the user's request the October 5 image-space glide (performance review
20261004, C3) is back, on top of vanilla steps: Civ III still chooses every
step and its timing, and the display slides between them. `C3X_RENDERER_GLIDE=1`
turns it on (off by default). The helper wire is version 17 (the presented
slide offset is back), and picking subtracts the presented offset
(`C3X_NATIVE_PAN_PRESENTED`, in `injected_code.c`). Unlike October 5, a glide
does not lift the in-job frame cap: with it lifted, light-save vertical steps
slipped to every other tick (153 ms, l65); with the cap, 80 ms (u66).
Test: `test_native_visual_cadence.py` `test_glides_keep_the_in_job_frame_cap`.

**Results.** Scroll fps: light save 57–59 with glide, 49–56 without (l65, l64);
3350 BC save 36–41 against 28–29 (u66, u67). Steps stay at native pace.

**Motion (3350 BC save, 1× scroll, trace level 2, u68 against u69).** Without
glide, 117 of 184 presented frames do not move and the rest jump 128 px. With
glide, frames move 5–53 px (p10–p90, median 27), so motion is continuous. The
display trails Civ III's camera by one to two steps (the presented offset
stays at 55–245 px), so it coasts briefly after a scroll stops.

**What still keeps it from feeling like Civ VI: frame pacing.** Frames arrive
in bursts (several within 10 ms, then 40–60 ms gaps; gap p50 19 ms, p90
43 ms). Stepped motion hides this; continuous motion shows it as judder. Even
frame pacing during motion (presentation stalls and per-frame cost, G2 and
G3) is the next requirement, not the glide itself.

**Environment.** Since about 13:50 a Windows activation dialog in the VM holds
focus, so scripted wheel input reaches no game window and zoom segments
register no target. Scroll and click input are unaffected. (Closed at about
16:00; l72 zoom is valid.)

## 18. Where the glide's frame gaps come from (October 8)

**Tried: presenting through native interface redraws while moving.** The
helper pauses frames while a native image batch runs. With a glide or zoom
moving, it kept presenting the last committed front instead
(`C3X_NATIVE_VISUAL_MOTION`, op 138). No measurable change, glide on:
- 3350 BC save, 1× scroll: 41.1 fps, 85 of 246 frame gaps over 30 ms (u70:
  92), gap p50 19.7 ms, p90 46.8 ms (u71).
- Light save: 1× scroll 57.2 fps, 2× 42.3, 3× 45.9 (l72).

**Cause: the camera job.** Each scroll step's camera job (about 21 ms on the
3350 BC save) runs on the helper's job thread. Frames during a job come only
from the job's own checkpoints, at most 30 Hz (the in-job cap, section 11), and
the first frame after the job waits for that cap. A typical gap: present at
0 ms, job starts at 2, completes at 23, next present at 44. So scroll pacing is
bounded by the camera job, not by interface batches. Lifting the cap is not
the fix (it slowed light-save steps to 153 ms, section 17). Rendering every
frame from resident content at the camera to be shown, with no camera job in
the frame path, is: the resident-world stage in the
[camera decoupling design](camera_decoupling_design.md).

**Other results from the same runs.**
- Light-save zoom (l72): starts 27–75 ms after the wheel, settles in
  154–280 ms, 21–56 fps during transitions, frame gaps up to 93 ms. G2 is not
  met.
- Minimap jumps reach Civ III's camera in 41 and 97 ms (light) and 62 and
  83 ms (3350 BC). The busy save took 1.6–2.4 s (section 15).

## 19. What a camera step redoes, and memory (October 8)

**Busy-save camera step, by phase** (b58, 74 jobs, trace level 2). The job
takes 86 ms at the median (163 ms at p90):
- the scene draw at the new camera: 42 ms;
- tile uploads: 16 ms (p90 52);
- set-up: 5 ms;
- unit selection: 5 ms;
- the rest: small.

At an unchanged camera the same draw prepares in about 1 ms. The step-only
costs are camera-dependent caches, rebuilt or re-checked over their whole
area rather than only the new edge:
- **Shadow pages: 17 ms.** About 8 of 25 pages are redrawn per step. Most of
  the time is bookkeeping over every caster (collecting 2.8 ms, proofs
  5.6 ms, selection 2.1 ms); the page draws are 3 ms. Pages keep the
  maximum height, so adding casters to a page is exact without clearing it.
- **Static layer: 11 ms.**
  - Each step changes the resident tile set, which forces a full membership
    proof over the whole layer.
  - Its guard band (320 × 192 px) reaches past the resident tiles (about
    288 × 172 px plus tall-object reach), so tiles entering it cause repairs
    of 15–21% of the layer.
  - It recentres every 2–3 steps (18–28 ms).
- **Tile uploads.** The whole world is already prepared and held compressed
  in RAM (16,900 records, 780 MB; 2.3 GB uncompressed). A step restores
  70–180 records from it and uploads them; nothing is compiled from scratch.

On the 3350 BC save every explored tile stays resident (199 reused, none
built), yet a step still takes 21 ms, spread over many small phases. Keeping
the world loaded removes only the upload share; the camera-dependent caches
are the rest.

**Memory census** (m1, busy save, `near`, `tools/sample_memory.ps1` every 2 s,
`C3X_RENDERER_PROFILE=1` with `C3X_RENDERER_MEMORY_CENSUS=1`):

| | Process memory | GPU memory (shared) |
| --- | --- | --- |
| Civ III | 304 MB median, 315 MB max | 21–41 MB |
| Renderer64 | 6.8 GB median, 7.2 GB max | 3.0 GB median, 3.9 GB max |

- **Tile geometry:** 0.8–1.35 GB resident on the GPU, out of 2.0 GB for the
  whole world; the in-RAM world copy is 780 MB.
- **The other ~2 GB of GPU memory** is render targets, textures and caches.
  Not yet broken down. The static layer alone can reach about 0.5 GB:
  - 4 slots plus 4 retained water-overlay layers, each about 55 MB at this
    resolution;
  - 2 preview images of the same size.
- **Process memory** includes the driver's copies of GPU resources.
- Civ VI asks for about 4 GB RAM and 1 GB GPU memory (minimum), 8 GB and
  2 GB (recommended).
- The user set Civ VI parity as a goal (performance goals, G1).

## 20. Memory by owner (October 8)

**Method.** `memory-census` (`C3X_RENDERER_MEMORY_CENSUS=1`) every 5 s during
camera jobs. Textures and targets are sized from their D3D11 descriptions
(`render_core/resource_census.h`, each resource once). Owners with their own
ledgers report those. `tools/sample_memory.ps1` samples process and GPU
memory every 2 s alongside. Runs: `near`, m2 (busy 1498 AD) and m3 (3350 BC).
DXGI's adapter usage in the VM moves between 1.0 and 2.4 GB while the owners
hold 3 GB, so totals come from Windows' per-process GPU counters.

**Renderer64 GPU memory by owner (MB):**

| Owner | Busy (median / max) | 3350 BC |
| --- | --- | --- |
| Tile geometry | 905 / 1,212 | 90 |
| Units | 597 | 597 |
| Static layer: slots | 105 / 209 | 209 |
| Static layer: water overlays | 105 / 209 | 209 |
| Static layer: previews | 52 / 105 | 52 |
| Static layer: view copy | 31 | 31 |
| Native UI composition | 323 / 366 | 359 / 388 |
| Terrain textures | 261 | 261 |
| Shadow atlas | 228 | 228 |
| Scene targets | 93 | 93 |
| Instances | 33 / 49 | 1 |
| Owners total | about 3,000 at most | about 2,130 at most |
| Windows GPU total | 2,362 / 3,947 | 2,661 / 2,758 |
| Process memory | 6,798 / 7,062 | 4,566 / 4,731 |

**Findings.**
- **Fixed costs dominate.** On the small 3350 BC map, geometry is 90 MB, but
  the renderer still holds 2.7 GB of GPU memory and 4.6 GB of process memory.
- **Units: all 78 unit types are loaded at start** (626 actions, 2,648
  meshes, 317 textures), whatever is on the map. That is 1.25 GB of process
  memory and about 0.6 GB of GPU memory; the 3350 BC save has 5 units.
- **The static layer** holds up to 0.55 GB: two zoom lanes with front and
  back slots, a retained water-overlay layer per slot, and two preview images.
- **Native UI composition** holds 0.32–0.39 GB.
- **The shadow atlas** is 228 MB.
- **About 0.6–0.9 GB is not yet attributed**, likely natural-world and object
  textures, swap chains and transient uploads.
- **Process memory** is the GPU memory plus the driver's copies, the in-RAM
  world (780 MB busy) and the unit sources (1.25 GB).

## 21. Unit types on demand (October 8)

**Change.** Game load no longer pins all 78 unit types
(`prepare_known_unit_sources({},false,…)`). Frames already loaded the units
they draw, on 4 workers with least-recently-used eviction. Each camera job
now also offers every captured unit's missing meshes and textures to idle
workers (`warm_unit_assets`). The capture reaches about 2.8 views, so a type is
decoded before its unit enters the view. `C3X_RENDERER_UNIT_PRELOAD=1`
restores the preload for A/B runs. Test: `test_unit_assets_on_demand.py`.

**Memory (3350 BC save, u4 against u5, same build).**

| | Process memory (median / max) | GPU memory (median / max) | Unit assets |
| --- | --- | --- | --- |
| On demand | 2,641 / 4,473 MB | 1,758 / 2,420 MB | 67 MB |
| Preload | 4,359 / 4,703 MB | 2,420 / 2,670 MB | 597 MB |

All on-demand loading happened in the first frame after load (about 0.33 s
over 12 turns), before input. None ran during scrolling, zoom or jumps.

**Busy save (b6).** Unit assets fell from 597 to 94 MB, with no loading during
play (34 turns, all before input). Process and GPU memory did not fall
(6.3 GB and 3.4 GB median): the freed memory went to tile geometry, whose
budget follows free memory (2.75 GB). The whole busy world (1.98 GB of
geometry) became GPU-resident; before, about half fit.

**Frame rate and steps (same build, `-MeasureCadence`, on demand against
preload).**

| Segment | Busy, run 1 (b7 / b8) | Busy, run 2 (b9 / b10) | 3350 BC (u6 / u7) |
| --- | --- | --- | --- |
| 1× idle fps | 45.1 / 41.8 | 45.2 / 33.5 | 60.0 / 60.0 |
| 1× scroll x, px/s | 405 / 341 | 446 / 320 | 2,648 / 2,634 |
| 1× scroll fps | 9.6 / 6.2 | 7.9 / 6.4 | 36.0 / 32.7 |
| Zoom in to 2×, fps | 16.7 / 26.8 | 33.7 / 15.4 | 50.4 / 51.1 |
| 2× scroll, px/s | 332 / 256 | 333 / 282 | 2,937 / 2,947 (35.6 / 32.0 fps) |
| 3× scroll, px/s | 354 / 201 | 319 / 236 | 1,709 / 1,650 (34.6 / 31.5 fps) |
| Jumps, ms | 628, 826 / 953, 985 | 559, 780 / 1,414, 964 | 104 / 96 |

The zoom-in difference reversed between runs: it is noise. On the busy save,
keeping the whole world resident speeds up scroll steps and jumps. On the
3350 BC save, scroll runs about 10% faster and nothing else changes.

## 22. Zoom-lane release and the unused shadow field (October 8)

**Changes.**
- **Zoom lane.** The static layer releases the zoom lane (slots 2 and 3 with
  their water overlays) and both preview images after 10 s at a complete,
  settled 1× layer (`release_zoom_lane`, traced as
  `static-zoom-lane-released`). The next zoom re-creates them, as the first
  zoom of a session does. Test: `test_static_zoom_lane_release.py`.
- **Shadow field.** The production shadow page field (32 slices of
  1024 × 1024 R32F, 128 MB) is created on first use (`ensure_field`).
  Renderer64's scene samples its own 100 MB atlas and never draws it. Test:
  `test_source_shadow_lazy_field.py`.

**3350 BC save, all changes so far (u8 memory run, u9 frame rate):**

| | Process memory (median / max) | GPU memory (median / max) |
| --- | --- | --- |
| Section 20 baseline (m3) | 4,566 / 4,731 MB | 2,661 / 2,758 MB |
| Units on demand (u4) | 2,641 / 4,473 MB | 1,758 / 2,420 MB |
| Plus zoom lane and shadow field (u8) | 2,481 / 2,811 MB | 1,631 / 1,910 MB |

The shadow category fell from 228 to 100 MB. Frame rate (u9) is within run
noise of u6:
- 1× idle and the 1× idle end: 60 fps;
- scroll: 31.8 fps (1×), 34.9 (2×), 35.1 (3×), at native step pace;
- zoom in: 51.6 fps;
- jumps: 128 and 115 ms.

## 23. Busy camera steps and frames after the memory changes (b11, October 8)

Trace level 2, route witness, busy save, `near`.

**Camera step job: 84.5 ms median, 226 ms at p90** (b58: 86 ms).
- Tile uploads fell from 15.7 to 3.9 ms at the median, since the world is
  resident. The p90 is still 57 ms.
- The scene draw at the new camera is still 40 ms:
  - shadows 17.3 ms (refresh 3.5, proofs 5.3, caster selection 1.9, page
    draws 2.7);
  - static layer 11 ms;
  - water 3.9 ms; reflection 2.7 ms; units 2.1 ms.

**Presented frames:** 18 ms median, 35 ms at p90.
- Scene re-render for animation (`compose_prepare`): 10.1 ms.
- Evaluation of about 2,560 native interface operations: 2.0 ms.
- Heavy frames reach 5–21 ms of evaluation and up to 26 ms of scene
  preparation.

## 24. Glide fixes, and the glide on by default (October 8)

**Bug (user report, 3350 BC save, 2× and 3× scroll).** With the glide on,
zoomed scrolling showed repeated vertical slices or black at the trailing
screen edge. Window frames at 10 Hz (v1–v7, `near`) and per-frame slide
offsets (v8, route witness) found two causes:
- **Nested trailing strips.** Each new step copied the selected world's
  output as the previous world. While a slide ran, that output was the slid
  composite, whose trailing strip was itself an older world, so continuous
  scrolling nested the strips. The trailing world is now copied from the
  selection's inputs at the start of Civ III's next world pass
  (`copy_world_inputs`).
- **Zoom.** A step whose trailing world was copied at another zoom does not
  start, and camera moves across a zoom change (Civ III re-anchoring the
  camera) do not slide.

A first attempt removed the zoom scaling of steps; it was wrong (the
selected world is the zoomed view, and Civ III's native steps are 128, 128
and 84 px at 1×, 2× and 3×) and was reverted. Tests: `test_pan_transition.py`.

**Verification.** A seam detector (an abrupt column change within 400 px of
either edge) over every frame of 1×–3× scroll: no seams in 2× and 3×
scroll. Slides occur only during scroll segments and the coast after them
(v8: 187 of 196 presents in 2× scroll, 186 of 198 in 3×).
- Seams remain in some zoom transitions in both glide runs (14 frames during
  zoom-in notches, 5–9 after the 3× zoom), but no slide is active there. One
  glide-off run had none. This is an intermittent zoom preview issue for G2,
  not the glide.

**Default.** At the user's request the glide is on by default;
`C3X_RENDERER_GLIDE=0` restores stepped display.

## 25. Baseline for the architecture plan (October 8, evening)

The staged plan toward Civ VI's architecture (stages 0–6,
[camera decoupling design](camera_decoupling_design.md)) is measured against
this baseline. Build: commit f12529cb plus per-part static-layer timers
(trace level 2 only). The uncommitted experiment (a grow-only shadow span and
paused static refinement while moving) was reverted: its runs (67 and 76 ms
against 84.5 ms per busy step job) were within run noise, the pause fired on 5
of 58 step frames, and stage 1 replaces both with world-anchored caches.

New tools: `tools/seam_report.py` (straight seams in 10 Hz window frames per
`near` segment; calibrated on three real busy-save stale strips and 87 clean
frames) and `tools/frame_gap_report.py` (presented-frame gaps per segment from
route-presented traces).

**Runs** (`near`; a0, a2, a4 `-MeasureCadence -SampleHz 10`; a1, a3, a5 trace
level 2 with route witness and unbuffered traces; a6 memory census).

| Segment | Busy 1498 AD (a0) | User 3350 BC (a2) | Light (a4) |
| --- | --- | --- | --- |
| 1× idle | 36.7 fps | 59.9 fps | 60.0 fps |
| 1× scroll x | 364 px/s, 10.6 fps, step 313 ms | 2,565 px/s, 41.9 fps, step 79 ms | 2,786 px/s, 57.6 fps, step 78 ms |
| 2× scroll | 231 px/s, 9.3 fps, step 609 ms | 2,587 px/s, 35.2 fps | 2,937 px/s, 51.4 fps |
| 3× scroll | 168 px/s, 9.0 fps, step 543 ms | 2,262 px/s, 37.3 fps | 1,702 px/s, 50.4 fps |
| 3× idle | 45.0 fps | 59.7 fps | 60.3 fps |
| Zoom in / out | 27.5 / 12.1 fps | 50.4 / 51.4 fps | 53.9 / 50.8 fps |
| Zoom start (wheel to motion) | 157–1,819 ms | 27–45 ms | 29–47 ms |
| Jumps | 872, 861 ms | 85 ms | 73, 78 ms |

**Frame gaps while scrolling** (route-presented, a1, a3, a5):

| | Busy | User | Light |
| --- | --- | --- | --- |
| 1× scroll x, p50 / p90 | 58 / 127 ms | 17 / 43 ms | 22 / 37 ms |
| 1× scroll x, gaps over 30 ms | 61 of 85 | 82 of 267 | 77 of 280 |
| 2× scroll, p90 | 217 ms | 49 ms | 34 ms |
| 3× scroll, p90 | 227 ms | 47 ms | 34 ms |

Even the light save's 10 ms step job leaves a quarter of scroll frame gaps
over 30 ms: frames during a job come only from its checkpoints (section 18).

**Camera step jobs** (`step_report.py`, p50): busy 79.7 ms (shadows 15.2,
scene preparation 18.9, static 10.9, reflection 2.3; queued 95 ms; request to
adoption 171 ms); user 17.6 ms (adoption 61 ms); light 10.4 ms (56.5 ms).

**What the busy step's static and shadow work is** (a1, 53 step frames, 58
shadow builds):
- On 30 step frames the displayed static layer was stale because the shadow
  sampling identity changed (key word 3, a span refit): the step showed a
  preview while a back slot refined (static 7 ms). 8 shadow builds redrew all
  25 pages.
- On 18 step frames tiles entering residency caused repairs of 12–26 ms.
- Every step re-proved membership over the whole layer (about 3 ms).
- Recentring was rare and cheap (8 frames, mean 0.4 ms, p90 2.4 ms); section
  19's recentre cost no longer applies after the memory changes.
- Shadow builds: 46 of 58 were triggered only by a residency change
  (`prepared_signature`), 11 by the span. Medians: caster refresh 3.1, proofs
  4.6, caster selection 1.9, page draws 3.0 ms.

So stage 1 starts with the sampling span (fixed per zoom step), then
residency-independent static coverage and proofs, then caster deltas; the
wrap-around slot is not needed for cost and is deferred until stage 2 needs
it.

**Memory** (a6, busy, `sample_memory.ps1` and census): Renderer64 6,161 MB
process memory median (6,670 max) and 3,359 MB GPU (4,317 max); Civ III 371 MB.
Census: geometry 2,300 MB, terrain textures 261, static slots 105 plus
overlays 105, static cache 31, shadow 100, scene targets 93, units 94,
instances 52, composition 53. Geometry is 72% of the attributed GPU memory:
stage 5's main target.

**Seams** (`seam_report.py`, a0, a2, a4): during scroll, busy 16 of 93
frames (plus one coast frame), user 35 of 146, light 6 of 146 (weak). All
inspected cases are stale strips at the trailing edge. Cause: a step that
arrives before the previous slide finished starts from the unfinished offset
plus the step; the trailing world covered only one step, so a strip as wide
as the leftover was never written (`RetainedComposition`, panning branch).
Busy steps arrive every 230–750 ms and the user save's 78 ms steps are
shorter than the 150 ms minimum cruise, so overlaps are routine. Section 24's
check did not catch them.

**Failure under profiling.** a1 hit "renderer publication pressure"
(65,536 work items) 3 s after the last zoom-out notch with trace level 2 and
unbuffered traces; a0 (same build, trace 0) was clean. Scroll segments
precede it.

## 26. Glide strip fix and stage 1, first measurements (October 8, night)

**Glide trailing strip.** The trailing image for a new step is now the
selected world composed as the next frame would show it (the committed world at
its slide offset over that slide's own trailing image), and the slide places
it by that offset (`RetainedComposition::copy_world_output`,
`PanTransition::begin`'s copied offset). The resting world alone left a stale
strip as wide as the unfinished offset; the last drawn frame was a step behind
whenever no frame ran between a commit and the next step. Test:
`test_pan_transition.py` (a one-row model driven by the real `PanTransition`;
the old rules leave stale or misplaced pixels). User save (g2 against a2):
seam frames in 1×–3× scroll 35 → 4. The remaining four follow a long camera
move that was slid as one step (the view jumped from a city to unexplored
land within one step); stage 4 removes trailing strips. Scroll fps and step
pace unchanged within noise (1× x 39.3 fps, 2,502 px/s; 2× 36.1; 3× 36.5).
Busy save (g6 against a0): 16 of 93 scroll frames → 12 of 110 (1× 11 → 2,
2× 3 → 2, 3× 2 → 8). The 3× seams follow presentations at a rendered zoom of
1.0 from inside camera jobs, interleaved with 3× frames (route-presented
`zoom_q16`); stage 2c removes camera jobs from steps.

**Stage 1a, shadow sampling span.** The span is fixed by the receiver region
(the region of interest at the shadow zoom) and the light, grows only when a
receiver reaches past it, and is remembered per region
(`StableShadowSpan`, `shadow_sampling_grid.h`; test
`test_stable_shadow_span.py`). Density: the new span is 0.92× the old rule's
median span (p10 0.88, p90 1.21), so shadows are as sharp or slightly sharper
on most frames.

Busy save, trace level 2 (g3, g5; a1 is the baseline):

| | a1 | g3 | g5 (remembered spans) |
| --- | --- | --- | --- |
| Span refits over the run | 11 | 36 | 20 |
| Step job p50 | 79.7 ms | 65.0 ms | 76.1 ms |
| Static on step frames, mean | 16.9 ms | 11.7 ms | 13.8 ms |
| Step frames with a stale static layer | 30 of 53 | 48 of 70 | 39 of 60 |

The step job did not change beyond run noise. Three findings explain why:
- **Zoom changes the receiver region**, so each zoom level starts a new span;
  the busy save's receivers also reach past the region estimate (light-space
  v extent 83 against 30–46), so each level grows once. Remembering spans per
  level halves the refits.
- **A stale static layer stays stale while scrolling.** Its back-slot
  refinement restarts whenever the camera leaves the back slot, so after a
  refit 13–15 consecutive steps show a preview. A world-anchored back slot
  (the wrap-around slot) is needed for refinement to survive camera motion;
  section 25's conclusion that it could wait was wrong.
- **Repairs come from residency churn, mostly shadow casters.** Repair causes
  per 1× y step (g5): about 150 shadow footprints of casters entering or leaving
  the resident set and about 44 new contributor keys; no visibility, proof or
  order causes. A 4-tile input ring (g4) did not reduce them. Civ III's capture
  defines residency around each step's camera (appearance halo of 8 tile
  coordinates: 512 px in x, 256 px in y), which is not enough room for a
  selection that is both stable and covers the static guard band plus shadow
  reach.

Also: at 2× and 3× every step's job renders the hidden canonical 1× lane, and
its static layer is stale there (a different shadow region), costing static
work on every zoomed step for a lane that is not displayed.

## 27. Stage 2a: a block-anchored resident world (October 8, night)

**Change (behind `C3X_RENDERER_WORLD_WINDOW=1`).** The camera job's resident
geometry is selected by a world window instead of Civ III's capture
(`render_core/world_window.h`). The window reaches 8 tile coordinates past the
view in x and 12 in y, its origin is snapped to 8×8-coordinate blocks and its
extent is fixed, so it changes once per block of camera motion (every four 1×
x steps, every two y steps). Tiles the capture lacks come from the retained
world (the record's last full copy, else a full permitted world input). The
window is listed in row-major order, so the membership diff sees a stable
order; Civ III still receives replacement flags in its own capture order
(`validate_custom_renderer_replacement_ownership` requires that). A block
crossing is an ordinary membership change (`ForegroundSelection::same_rule`),
not a full rebuild. Tests: `test_world_window.py`.

**Result (busy save, w3 against g5, trace level 2).**
- Membership: 17 of 39 camera jobs kept the resident set unchanged ("covered"),
  21 changed it incrementally, 1 rebuilt it. Zoomed scrolls and revisits are
  covered at every step. At 1× each step still adds 10–90 tiles at the leading
  edge: Civ III captures appearance only 8 coordinates ahead, and the
  topology-only band beyond it replaces the full world input of tiles never
  captured with appearance, so the window cannot draw them until they come
  within the appearance halo.
- Step job p50 84.5 ms (g5 76.1); request to adoption 242 ms (170). With
  `C3X_RENDERER_WORLD_WINDOW=1` and no other change, busy scrolling is slower:
  the resident set is about twice as large, and a first version that rebuilt
  it at every block crossing took 0.8–1.1 s per step (w1).
- **A covered step still costs about 45 ms**: scene preparation 33 ms (shadows
  with city lights 17–18.5, visibility selection 8–10, region-of-interest body
  requirements 4.2–4.4, setup 1.5–2) and static 12.5 ms. An unchanged camera
  prepares in about 1 ms. So most per-step cost comes from computations keyed
  to the camera position, not to residency: the shadow page window and its
  caster bookkeeping, the per-view visibility selection and the 128 px
  region of interest.

The window stays off by default. Stable residency is a prerequisite for making
those per-camera computations incremental; that is the next piece of stage 2.

## 28. Water visibility from the retained world (October 9)

**Cause.** Water and river records carry `water_visible`, refreshed from the
current frame's fog coverage. Coverage spans only the captured view, so water
off it read as hidden and flipped as the camera moved; each flip edited the
resident set, which gave it a new revision, and everything keyed to that
revision (shadow caster bookkeeping, static proofs, visibility selection,
region of interest, unit plans) rebuilt. With the world window the camera job
(window tiles) and visual frames (Civ III's capture) disagreed about the
coverage on every step, so even an unchanged resident set got two new
revisions per step. The flag now follows each tile's retained visibility
(`topology_cache.retained(...)->visibility_flags`, keyed by
`visibility_sequence()`), which both agree on. Tests: `test_world_window.py`
(`WorldWaterVisibilityTests`), `test_visibility.py` (harness follows the
retained record).

**Result (busy save, trace level 2, w4 with the window, w5 without).**
- With the window, an unchanged resident set now keeps its revision. On those
  steps (25 of 65 jobs): scene preparation 6.3 ms (33 before), static 6.1 ms
  (12.5), and the shadow atlas was reused on 22 of 25.
- Steps that change the resident set still cost prepare 26.8 ms, static 11.6,
  shadows with a caster refresh (5.9 ms) and proofs (6.7 ms).
- Step job p50: 80.6 ms with the window, 83.2 ms without. At 1× each step
  still adds about 48 tiles at the leading edge (the appearance halo limit,
  section 27), so most 1× steps are not covered yet.
- w5 hit the publication-pressure failure again under trace level 2 with
  unbuffered traces (as a1); runs at trace level 0 have not.

## 29. The window's leading edge needs Civ III's capture (October 9)

**Why tiles kept entering.** A `world-window` trace (trace level 1) counts
window tiles the retained world cannot supply. On the busy save 600–1,000 per
step had no full appearance: their records hold no full copy and their world
inputs are the seed's minimal fog records. The world seed sends full records
only for visible tiles; explored fog gets terrain, city body, routes and
remembered overlays, deliberately without live resources, territory or
buildings (`read_custom_renderer_world_record`). The per-step capture reads
those live facts for every explored tile it covers
(`read_custom_renderer_tile`), so the two cannot stand in for each other.

**The capture envelope.** With custom zoom, Civ III's per-step capture covers
`custom_renderer_capture_bounds` (about 4 tile coordinates past the visible
tiles at 1×) and skips the halo. Behind `C3X_RENDERER_WORLD_WINDOW=1` the
tile capture (not the unit capture) now reaches 16 coordinates in x and 20 in
y past the visible tiles (`injected_code.c`, `capture_custom_renderer_topology`;
the halo path got the same per-axis reach). Installed with `INSTALL.bat`.

**Result (busy save, w8 against w4).**
- The window is complete: 0–108 tiles skipped per step, none synthesized.
  33 of 62 jobs kept the resident set; it changes only at block crossings (one
  band in, one out).
- Step job p50 60.8 ms (80.6): shadow builds 0 ms at the median, scene
  preparation 13.5 ms, static 6.0 ms.
- But Civ III's native map pass after each step rose from 16 to 61 ms
  (median, 1× scroll), and request to adoption from 172 to 221 ms. Not yet
  attributed. (A first reading blamed the publication backlog; the
  `native-call-waits backlog_ms` figure measures the bridge's input
  hit-test queue, not tile publication.) Each step copies the full tile
  array (512 bytes per tile) three times on Civ III's thread, compares it
  byte for byte, re-encodes it for the helper and decodes a full-frame
  completion reply, all of which scale with the envelope.

The capture protocol is camera-anchored: each step re-sends every tile of the
envelope, changed or not. The incremental design keeps that logic in the
bridge and helper (injected code only asks the bridge for the capture margin,
`C3X_NATIVE_CAPTURE_MARGIN`), after measuring where Civ III's thread spends
the extra time.

## 30. Incremental capture, off-screen HUD draws and the glide's removal (October 9)

**Incremental capture.** Camera requests are now deltas
(`camera_delta.h`, subtype 6; `C3X_RENDERER_CAMERA_DELTA=0` restores full
frames). Each occurrence carries its coordinates and anchor; full fields only
for tiles whose content changed, the topology array only when it changed. The
helper keeps the tiles it has received and rebuilds the exact frame; a
receiver missing a referenced tile asks for a base. In the test fixture a
128 px step sends 32 KB instead of 275 KB (`test_camera_delta.py`). All of it
lives in the bridge and helper; injected code is unchanged.

**It was not the cost.** Busy save, 1× scroll in x, Civ III's native map pass
(median): 98 ms with the window (106 before, w11 against w9), 32 ms without
it (37, w12 against w10). Step job p50 61.5 ms with the window, 74.8 ms
without.

**The cost: off-screen HUD draws behind the bridge's hit-test queue.**
- With the window, Civ III's thread waited on the input hit-test queue for
  1.21 s in matched 1× windows, against 0.36 s without
  (`native-call-waits backlog_ms`). In one 115 ms step, about 85 ms were
  line and sprite draws stalled on that queue; the capture itself took 9 ms.
- Traced with `C3X_RENDERER_HIT_TRACE=1` (w13 with the window, w14 without):
  the same number of map redraws (17 against 15 in 1× y scroll), but 399
  fills per redraw on the map canvas against 89, mostly 24×22 box outlines.
- The capture margin was captured as RENDER tiles, so the injected HUD pass
  (`patch_Main_Screen_Form_draw_city_hud`) drew a unit status box for every
  off-screen unit in the margin, about 100 boxes per redraw instead of 22.
- **Fix.** Tiles past the zoom envelope are appearance only, as in the halo
  (visibility bits, `TOPOLOGY_HALO | PREFETCH`). The window reports
  replacement ownership only for tiles Civ III captured as RENDER
  (`WorldWindow::native_flags`). Tests: `test_world_window.py`
  (`CaptureMarginTests`; prefetch tiles report no ownership).
- `hit_trace_report.py` now stops at a record the process cut off mid-write.
- **Result (busy save, no glide, w15 with the window, w16 without).**

  | | window, before (w11) | window (w15) | no window (w16) |
  |---|---|---|---|
  | native map pass after a 1× x step | 98 ms | 52 ms | 31 ms |
  | after a 1× y step | 50 ms | 16 ms | 12 ms |
  | after a 2× / 3× step | 87 / 87 ms | 27 / 26 ms | 40 / 24 ms |
  | native pass, step median | 83.6 ms | 24.0 ms | 23.0 ms |
  | step job p50 | 61.5 ms | 82.8 ms | 91.5 ms |
  | request to adoption p50 | 233 ms | 198 ms | 172 ms |
  | resident set kept | 32 of 60 | 35 of 69 | 1 of 62 |

  With the window, Civ III's per-step pass now matches the native envelope
  except at 1× x (52 against 31 ms). Step jobs were slower in both of this
  pair (no window: 74.8 ms in w12, 91.5 ms in w16) outside the traced
  phases; not yet attributed (run variation or the glide's removal).
- **Where the window stands.** Jobs that keep the resident set take
  47.9 ms (p50; p90 101). Block crossings take 164 ms (p90 372), against
  93.8 ms for an ordinary step without the window, because the whole
  entering band (8 coordinates deep) is built and its shadows rebuilt inside
  the step's job. That band lies past the guard band, outside every drawn
  pixel, so a step need not wait for it: building it after the step is shown,
  in bounded slices, is stage 2c. Until then the window stays off by default.
- Nearly every tile a crossing builds is restored from the compressed RAM
  backing (101 of 102 in one 1× crossing), not compiled. Re-queuing the
  completed world-preparation regions of the next band (w17) changed
  nothing: background region preparation of a compiled region returns in
  1–3 ms with `built=0 reused=0` and does not bring its geometry back to the
  GPU. Reverted; restoring the next band to the GPU in idle slices is part
  of stage 2c.

**The scroll glide is removed (the user, October 9).** Each Civ III camera
step is shown at the camera Civ III chose, as in the vanilla game. Removed:
`PanTransition`, the trailing-world composition, the presented offset in the
helper wire (back to version 16) and `C3X_NATIVE_PAN_PRESENTED` with
picking's subtraction of it. A step is still adopted on Civ III's next tick
after its image is ready (the existing deferral); stage 2c is meant to make
that the next tick. Tests: `test_native_visual_cadence.py`
(`test_camera_steps_are_presented_as_vanilla_jumps`), and
`test_zoom_integration.py`: picking queries only the presented zoom.

User save (3350 BC), `near` at 10 Hz, n1 without the glide against g1 with
it:

| segment | steps (n1 / g1) | native px/s (n1 / g1) | adopt p50 (n1 / g1) |
|---|---|---|---|
| 1× scroll x | 67 / 26 | 2,625 / 1,150 | 58 / 62 ms |
| 2× scroll | 53 / 49 | 2,789 / 2,682 | 60 / 63 ms |
| 3× scroll | 56 / 46 | 1,670 / 1,503 | 61 / 62 ms |

Civ III now steps on every 78 ms tick at 1× (with the glide, 26 steps in
6 s; why it stepped less often was not isolated), the map scrolls 2.3× as
far in the same time at 1×, and the
seam check finds no seam frames in any segment (g1: 4). Jump 2 took 47 ms
(108). Distinct-frame fps during scroll is lower (28 against 50 at 1× x)
because the glide's slide frames no longer count as new frames.

Busy save (1498 AD), n2 without the glide against g6 with it: 1× x 19 steps
in 6 s (16), adoption p50 238 ms (305); 2× 20 steps (16), 193 ms (283); 3×
21 steps (18), 199 ms (213); 1× y unchanged (15 steps). Idle 1× 38.8 fps
(38.6). Jumps 673 and 881 ms (965, 868). No seam frames. Busy steps still
come about four times a second against Civ III's 12.8 ticks: this is the
stage 2 reference.

## 31. Window coverage and crossing cost (October 9)

**Capture margin.** The window's fixed extent puts its leading edge up to
reach + 2 blocks past the view (24 x, 28 y coordinates), but Civ III
captured only reach + 1 block. After each crossing the last one or two
columns were uncaptured and entered one column per step (36 tiles at 1×):
about half of all steps changed the resident set. The margin is now
`WorldWindow::margin_x/margin_y` (reach + 2 blocks); `test_world_window.py`
checks every camera phase at 1× and zoomed-out tiles. Busy save (w18, window
on): every changed job is now a block crossing, and 45 of 64 jobs keep the
resident set.

**Warming the next band (tried, reverted).** Background region preparation
stays in RAM backing once GPU geometry is near its budget (always, on the
busy save). Letting regions under the window's next band take GPU residency
did warm them (82 regions, 1,029 tiles; crossings restored about half as
many tiles), but:
- crossings with warm geometry still took 100–170 ms: shadow proof
  registration for the entering casters 40–76 ms, resource preparation
  15–20 ms, static repair 10–20 ms, all proportional to the entering band;
- the background preparation competed with Civ III's emulated threads in
  the VM: covered steps 57 ms (about 45), Civ III's native pass after a 1× x
  step 144 ms (52), with the hit-test queue backed up again.

A crossing's cost is spread over several subsystems, each proportional to
the entering band, and in the VM moving it to background threads costs Civ
III's thread time. Crossing p50 157 ms (164 before), p90 700 ms.

## 32. Where a busy display frame goes (stage 3 input, October 9)

Busy save, `near`, default settings (h0: profiled; h1:
`C3X_RENDERER_PROFILE=3`, which drains the GPU at every mark so each phase
is charged its own GPU time; drained times are inflated, proportions hold).

- **Worker CPU per display frame, 1× idle:** 17.5 ms (p50 of sampled
  frames): composition prepare 9.6 ms (it includes the 3D scene sample),
  evaluate 2.0 ms (about 2,800 re-run operations), display 0.2 ms, about
  5.7 ms elsewhere in the frame. Each frame copies 19.6 M pixels on the GPU.
  `route-frame-budget` compose p50: 18.6 ms at 1× idle, 13.5 ms at 3× idle.
- **GPU per frame without a scene redraw (drained):** about 28 ms, nearly
  all of it the retained replay of Civ III's map-dependent interface:
  before-image assembly for re-run operations 12 ms, re-run native images
  5.5 ms, expansions 3.7 ms, the HUD batch (spatial run and two base copies)
  4.3 ms, assemble and display 2 ms. With a scene redraw the 3D passes add
  water 20.5 ms, static 13.8 ms, units 5.1 ms and reflections 4.8 ms
  (drained).
- **Why it re-runs:** the map image is new on every frame, so every
  operation that reads it (the HUD batch, operations with map-dependent
  inputs, the world selection and the changed front fragments) runs again.
  The HUD batch's per-pixel cache (resolved against dependent pixels)
  applies only to single spatial runs, still pays both base copies, and is
  discarded on every recompile, including each zoom transition.

Stage 3 targets this replay: the interface compiled once per committed
front, with only map-dependent pixels evaluated against the scene in the
final composite.

## 33. Unchanged renders and the composition oracle (October 9)

**Unchanged renders.** A visual frame's map sample returned the completed
render as a new image even when the scene had not been re-rendered, and the
projected view and projected map overlays re-ran unconditionally, so every
operation over the map ran again on every frame. A sample now carries its
render generation: the same generation at the same projection keeps the
imported image, the projected overlays and the projected view, so nothing
composed over them re-runs. On the busy save the units animate, so the
scene is re-rendered on every presented frame (h3: 382 renders for 382
presented frames in the last idle segment) and the interface replay is
unchanged; the saving applies to static scenes. Cadence n3 against n2: 1×
idle 39.0 fps (38.8), last idle segment 42.0 (39.0), scroll unchanged, no
seam frames.

**The composition oracle runs again.** `test_retained_composition.py` had
been failing before its first case: the single test function's frame
exceeds the default 1 MB stack (an access violation in the 32-bit build, a
stack overflow in 64-bit). It now runs on a thread with a 64 MB stack
reservation; all 664 GPU oracles pass, plus a new case for the unchanged
generation that fails on the old code.

Per frame on the busy save the interface must therefore be recomposed over
a new scene: stage 3 makes that recomposition cheap rather than rare.

## 34. What stage 3 can gain (October 9)

Cost attribution only: `C3X_RENDERER_DIAG_UI_HOLD=1` recomposes the
interface on one frame in eight and otherwise renders the scene and displays
the previous composite (a wrong picture; zoom transitions are not shown).
Busy save, `near` at 10 Hz (d1 against n3): 1× idle 55.8 fps (39.0), last
1× idle segment 59.5 fps (42.0), 3× idle 51.1 fps (47.2). The interface
replay costs about 7–8 ms of a 25.6 ms busy idle frame; removing most of it
reaches the G3 target (55 fps busy idle) without other changes. Scroll steps
and adoption are unchanged (18 steps at 1× x), as expected: steps wait on
camera jobs, not on frame cost alone.

## 35. Stage 3, first piece: the fused interface pass (October 9)

**Shape.** On the busy save the committed screen is Civ III's interface
canvas keyed over the selected world (a full-screen native image), the world
is one compiled HUD program (about 1,970 draws) over the projected view of
the scene, and a few small buttons are drawn after the transfer.

**Change.** That keyed transfer is evaluated in one compute pass
(`RetainedComposition::evaluate_fused`, `SpatialComposition::fused`): each
pixel starts from the projected scene, takes the view's native word (the view
transform's ordered quantization), runs its 32-pixel tile's HUD program in
registers and applies the keyed transfer, writing the node's two planes. The
view transform, both HUD base copies, the HUD dispatch, the world selection
and the transfer's before-image are gone. The HUD program is bound exactly as
its node binds it; unchanged inputs keep the result; anything that does not
match the shape (cross-position reads, interpreter runs, other formats or
extents) takes the general path. `C3X_RENDERER_FUSED_INTERFACE=0` turns it
off.

**Exactness.** `test_retained_composition.py` (`fused interface`): equal to
the live interpreter over six frames with a new scene each frame, with a
40-sprite HUD, an uploaded interface canvas and a small button drawn over
the transfer, for both native formats; equal to the general retained
evaluation at 1.5× and 3× zoom. The case fails on the old code. All 680
oracles, the fullscreen HUD recipe and the spatial composition tests pass.

**A bug found by measurement.** The first in-game run reported zoom 1.0 in
every segment: the presented scale (read by picking and Civ III's zoom
adoption) came from the view node, which the fused pass no longer evaluates.
The pass now sets it from the view; the test compares it with the general
path and fails without the fix.

**Result (busy save, f5 with the scale fix, against n3 and the d1 ceiling).**
The first run (f4) reported a stale zoom, so its zoomed segments are not
used.

| | fused (f5) | before (n3) | ceiling (d1) |
|---|---|---|---|
| 1× idle | 41.7 fps | 39.0 | 55.8 |
| 3× idle | 50.9 fps | 47.2 | 51.1 |
| last 1× idle segment | 47.9 fps | 42.0 | 59.5 |
| 1× scroll x | 19 steps, adoption 241 ms | 18, 227 ms | – |
| 3× scroll | 22 steps, adoption 137 ms | 21, 193 ms | – |

At 3× the native distance per step was larger than in n3 (1,989 against
354 native px/s); not yet explained. The fused pass ran on every sampled frame (6,376 fused evaluations, no
fallback) and no frame re-ran the interface's operations. No seam frames.
Still per frame: the projected scene assembly, the interface canvas's
assembly (77 parts), a full-screen quantization of the scene feeding the
bottom-right panel's blends, the small buttons, the front assembly copy and
display.

## 36. Stage 3: retained source canvases and the front write (October 9)

**Change.** The fused pass's two source canvases (Civ III's interface words
and color) persist between frames; a fragment is copied again only where the
previous assembly did not already take its pixels from the same node, output
and revision (a CPU comparison of about 77 rectangles, no pixel diff). When
the transfer's color plane is part of the displayed front, the pass also
writes the retained front texture, and the front assembly copies only the
other fragments. Cost: two persistent full-screen textures (about 21 MB at
2240×1192) inside the composition's fixed budget; a full interface redraw
still copies once. Tests (`fused interface`): source copies at most an
eighth of the screen per frame after the first; on animation-only frames
the front assembly copies only the button and each frame equals the general
retained evaluation.

**Result (busy save, f6 against f5, one run each).** 1× idle 43.7 fps
(41.7), 3× idle 53.6 (50.9), last 1× idle segment 42.7 (47.9). Same-build
runs vary by 3–5 fps per segment on this VM, so the effect of this change
is within run variation, consistent with an estimated 1–2 ms of GPU copies
per frame. Against the pre-stage-3 run (n3): 1× idle 39.0 → 43.7, 3× idle
47.2 → 53.6, last 1× idle segment 42.0 → 42.7.

## 37. Stage 3: the HUD cache in the fused pass; where stage 3 stands (October 9)

**Change.** The fused pass uses the HUD program's per-pixel cache as the HUD
pass did: the first run classifies each touched pixel, and pixels proven
independent of the map keep their cached result instead of re-running their
tile's program (`test_spatial_composition.py`: the fused pass classifies the
same 28,900 pixels as the HUD pass).

**Result (busy save, two runs f7a / f7b).** 1× idle 43.9 / 45.1 fps, 3× idle
52.5 / 50.8, last 1× idle segment 47.9 / 48.0. The two runs agree within
about 1.5 fps.

**Stage 3 so far** against the pre-stage-3 run (n3) and the ceiling with the
interface replay skipped (d1, section 34):

| | now (f7, mean) | before (n3) | ceiling (d1) |
|---|---|---|---|
| 1× idle | 44.5 fps | 39.0 | 55.8 |
| 3× idle | 51.7 fps | 47.2 | 51.1 |
| last 1× idle segment | 48.0 fps | 42.0 | 59.5 |

3× idle is at its ceiling. At 1× (about 2,800 interface operations against
1,800 at 3×) 8–11 fps remain; the ceiling also skipped the scene import
(a full-screen pass applying the CAS detail filter) and the display of a new
composite. Moving that filter into the fused pass would save one full-screen
pass but cannot promise bit-identical floating-point results across two
shaders, so stage 3 stops here for now. Per frame the interface CPU is about
1.5 ms of evaluation; about 30 small map-dependent operations remain (tiny
copies, 74 k pixels per frame).

## 38. How a busy scroll step's period forms; steps drawn at adoption (stage 2c, October 9)

**Civ III's tick.** Edge scrolling runs on a 66 ms window timer (`Timer::activate`
→ `SetTimer`), about 78 ms at the default timer resolution. Each tick runs the
Animator update first, which polls and adopts a ready step and redraws, and
then `scroll_at_mouse`, which requests the next step. A step therefore needs
its request delivered and its work done before a later tick polls it. When a
tick's own frame work runs past 78 ms, the next tick fires as soon as it ends,
before any step requested at its end can be ready. So a step takes at least
two ticks unless that frame work is short.

**Where the period went (s0b, busy save, window on).** Median step period
301 / 200 / 278 / 231 ms (1× x, 1× y, 2×, 3×), 3–4 ticks:
- request to job ready 80–105 ms (delivery 15–27, job 45–75);
- wait for the next tick 47–56 ms;
- the adoption pass 15–48 ms;
- the rest of Civ III's frame before the next request 54–75 ms.

Civ III's thread spends 300–630 ms of every 2 s of busy scrolling inside the
bridge's hooks:
- unit draws about 60 µs each;
- about 1 ms per navigation call;
- up to 214 ms waiting for the click-test queue (`native-call-waits
  backlog_ms`).

Each step was also drawn twice: once by its job, and again by its adoption
(17–23 ms) so that a newer pose shown in between could not rewind.

**Change.** A step that keeps the resident window and its tile content
completes at request time, reporting Civ III's tile ownership, and is drawn
once, by its adoption. Crossings and content changes keep full jobs.
- Ownership comes from the last draw by tile content
  (`render_core/deferred_step.h`). It is taken before that draw clears it for
  tiles Civ III did not capture to draw (RENDER), and it ignores placement
  and per-capture authority bits (`CITY_BODY_KNOWN`,
  `NATIVE_OVERLAYS_KNOWN`, which come with the view but not the margin).
- The adoption compares what it reported with what it drew
  (`deferred-step-ownership`). A first version took the masked array; this
  check found 61–92 entering tiles per step reported as unowned (at the edge
  of the zoom-out envelope, off screen at 1×).
- Refusals are traced with their reason (`deferred-step-refused`).
- `C3X_RENDERER_DEFERRED_STEPS=0` keeps every step a full job.
- Tests: `test_deferred_step.py` (ownership follows content across
  captures; content, exploration and block changes refuse; the adoption
  draws before publishing; the reported flags are the unmasked ones).

**Result (same build, deferral off d7 against on d6).**

| | off (d7) | on (d6) | on: deferred steps | on: full jobs |
|---|---|---|---|---|
| 1× x | 379 ms | 229 | 199 | 487 |
| 1× y | 221 ms | 200 | 156 | 353 |
| 2× | 312 ms | 298 | 231 | 329 |
| 3× | 230 ms | 161 | 156 | 393 |

- 70 steps against 63 in the same scroll time.
- 56 of 81 steps were deferred, with no ownership disagreements and no
  native failures.
- Camera requests wait longer in the bridge's queue (p50 26 against 7 ms):
  the adoption's draw holds the helper worker while the next request
  arrives.
- Run-to-run noise is large: the deferral-off run was slower than s0b
  (379 against 301 ms at 1× x).
- An unprofiled cadence run with deferral on (d8) had step medians of
  223 / 234 / 232 / 234 ms. It had no seam frames in any segment and no
  native failures. Idle was 40.6 fps at 1×, 56.6 at 3× and 46.3 in the last
  1× segment. Frames presented while scrolling: 6.8 / 19.0 / 7.4 / 6.1 per
  second.

**What remains for one step per tick.**
- Deferred steps take 156–231 ms, two ticks. The adoption's draw now lands
  inside Civ III's tick, which waits behind it (its frame after the pass is
  58–132 ms).
- The adoption must not hold Civ III or the transport. The step should be
  drawn in the next display frame, with the display holding the previous
  frame until the new one is ready.
- Civ III's per-tick time in the hooks must fall.
- Block crossings (25–31% of steps, 330–490 ms) need their entering band's
  work spread across steps.

Baselines recorded before deletion (window off s0a / on s0b, median ms):

| | wait | queued | send | job | adopt |
|---|---|---|---|---|---|
| s0a (off) | 19.9 | 12.1 | 8.7 | 63.8 | 168.5 |
| s0b (on) | 26.1 | 0.2 | 18.3 | 56.4 | 175.6 |

## 39. Drawing the step outside its adoption (stage 2c.2, tried and reverted, October 9)

**Change tried.** The adoption only published the step and scheduled its
draw as a worker job. The job ran outside the call gate and serviced
commands at its checkpoints. The display held the previous frame until a
frame had sampled the step's image, using a two-level hold: a pending draw
skips the frame before evaluation; an unsampled image is evaluated but not
presented.

**Result (d9 against d6, same options).**
- Deferred steps: 207 / 163 / 215 / 206 ms against 199 / 156 / 231 / 156.
- 61 steps against 70 in the scroll segments.
- No failures or ownership disagreements.
- The adoption itself now finished before Civ III's pass ended, as intended.

**Why it did not help.**
- The step's draw is about 55 ms of worker time (p90 99) with few
  checkpoints.
- Civ III's image commands hold the call gate while they wait for those
  checkpoints, and the next camera request waits behind them. In one step
  it waited 85 ms in the bridge queue and then 128 ms in service.
- Moving the draw did not reduce the worker's serial work per step: the
  draw, the first display frame's re-render (13 ms) and compose, and Civ
  III's image batches. The change was reverted; the patch is kept for when
  the worker has headroom.

**Per-step worker cost against an ordinary display frame (d9, medians).**

| | step draw | display frame |
|---|---|---|
| pre-draw geometry (window, membership, waves, resources, setup) | 17 ms | none |
| scene prepare | 10.1 ms | 1.1 |
| of which unit body requirements / setup / unit poses | 4.3 / 1.65 / 0.9 | 0 / 0.28 / 0.68 |
| static layer | 5.4 ms | 0.03 |
| resource coverage | 1.4 ms | 0.03 |
| water | 2.95 ms | 1.69 |

About 35 ms of each covered step's draw re-prepares, for a new camera, a
scene whose content did not change.

**Next.** Make a covered step a frame-level camera update: carry the
per-camera preparation across steps by translation (window tiles, unit body
requirements, static placement, resource coverage), so a covered step costs
about one display frame (13–20 ms).

## 40. A covered step's setup before its draw (October 9)

`render-setup-phases` (trace level 2) times `render()` from entry to the
scene draw. Busy save, window on, deferred steps (e2), medians:

| phase | covered step drawn at adoption | crossing (full job) |
|---|---|---|
| window build and initialization | 1.4 ms | 1.7 |
| world sources | 0.0 | 0.1 |
| settings and memory budgets | 1.1 | 1.1 |
| membership diff | 3.2 | 6.6 |
| geometry build and ownership | 0.7 | 51.8 |
| ownership traces to waves | 1.3 | 1.5 |
| waves, unit selection and assets | 5.1 | 9.4 |

A covered step spends about 13 ms before its draw and about 31 ms in it
(section 39). The draw's prepare is 10 ms against 1 ms for a display frame;
static placement is 5.4 against 0.03. No single item exceeds about 5 ms. A
crossing's cost is its geometry build (52 ms) plus the shadow and static
work in its draw.

## 41. Stage 2c, bounded finish: region, proof carry, click-test backlog (October 9)

The user chose a bounded finish: the two largest per-step items and Civ III's
click-test waits, measured once.

**1. The region of interest follows world-window blocks.** It had snapped
the camera to 128 px, and every vanilla 1× step is 128 px. So each step
rebuilt the unit body requirements, the city lights and the shadow fits keyed
to the region. It now snaps to the window's 512 × 256 px blocks (at 1×), with
the same padding, so any camera in a block cell stays covered.
- Body requirements were rebuilt 0 times over 60 covered steps; before, they
  were rebuilt at every step, at 4.3 ms each.
- A covered step's scene prepare fell from 10.1 to 2.8–3.1 ms.
- Test: `test_fresh_shared_submission.py` (vanilla steps inside a block keep
  the region; a crossing rebuilds it).

**2. The static proof carries across scroll strips.** Each step's strips grew
the slot's covered rectangle, which is part of the proof's key, so every step
re-proved the whole slot (2–4 ms). When strips are drawn from current inputs
and nothing has changed since that frame's proof, the proof now carries to
the extended rectangle (`RasterContributors::carry`).
- Proof time per covered step: 0 ms. The run carried 5,602 times and ran 179
  full proofs (27 after key changes, 21 after ledger changes).
- Test: `test_static_dependency_reuse.py`.
- A covered step's static stage still takes about 5 ms outside its timed
  parts; this was not pursued.

**3. Click-test backlog bound: 512 → 4096 operations.** During busy
scrolling, Civ III's thread waited 0–318 ms per 2 s on the input-coverage
queue. With 4096 it never waited, and the slowest query took 0.02 ms (9
queries).
- The queue's dominant cost is its model of the map canvas: 4.2 of 6.7 s in a
  traced run.
- The map canvas cannot be exempted: the queried main-screen canvas depends
  on it through UI buttons blended over map backgrounds.
- Test: `test_hit_scene_fast_paths.py`.

**Results (busy save, window on, median step period).**

| | deferral off (d7) | 2c.1 (d6) | items 1+2 (a) | + backlog 4096 (b) |
|---|---|---|---|---|
| 1× x | 379 ms | 229 | 193 | 234 |
| 1× y | 221 | 200 | 158 | 157 |
| 2× | 312 | 298 | 221 | 198 |
| 3× | 230 | 161 | 164 | 162 |
| deferred steps (1× x / 1× y / 2× / 3×) | — | 199 / 156 / 231 / 156 | 163 / 146 / 158 / 156 | 160 / 113 / 167 / 156 |
| steps in the scroll segments | 63 | 70 | 78 | 80 |

**Unprofiled cadence run (f3, against 2c.1's d8).**
- Step medians: 191 / 175 / 234 / 168 ms against 223 / 234 / 232 / 234.
- 80 steps against 74.
- No seam frames in any segment and no native failures.
- Idle: 42.2 fps at 1×, 54.8 at 3× and 41.2 in the last 1× segment, against
  40.6 / 56.6 / 46.3, within run noise.
- Jumps: 648 and 984 ms.

**Where stage 2c ends.**
- Covered steps run at a steady two ticks (about 156 ms), with some single
  ticks at 1× y.
- Block crossings take 270–570 ms and are 25–30% of steps. Their cost is the
  entering band's geometry and shadow work (section 40).

## 42. Stage 4.1: the wheel request reaches the next frame (October 9)

**Baseline (busy save, default configuration, z0).**
- From the wheel to the first presented frame at a new zoom: 44–316 ms
  (`zoom_report.py`), typically 150–300.
- During a transition: 6–20 fps, with gaps up to 255 ms.
- Civ III handles the wheel within 1 ms.

**Cause found.** The zoom target travelled in the ordered native stream.
When a notch arrived, that stream held 300–600 ms of Civ III's queued
interface batches. The target took effect about 190 ms later (z1).

**Change.**
- The bridge sequences each request. It sends the request in the ordered
  stream as before (so recordings replay), and also publishes it at once in
  the shared wire (`requested_zoom`).
- The helper forwards a changed request before each display frame. For
  400 ms after a request, its frames are not held for batch receipts or
  publication pressure.
- The frame applies the request as it starts, retargeting without sampling
  the transition first. Before, that first frame did not move.
- Only a newer sequence retargets, so the late ordered copy never undoes a
  newer request.
- Tests:
  - `test_zoom_transition.py` (sequence rule; the first frame moves);
  - `test_zoom_integration.py` (sequenced, published and ordered; forwarded
    before frames);
  - `test_retained_composition.cpp` (session: the newest request wins).

**Result.**
- The request is applied 6–25 ms after the wheel (z3).
- Wheel to the first presented change did not improve: 29–317 ms (z4).
- The first frame at a new zoom takes 50–125 ms to compose and present in the
  VM: every world layer is re-projected at the new scale, and the scene is
  re-rendered. In one notch the scene was ready 17 ms into the frame, and the
  frame was presented 95 ms later.
- Zoom-changing frames compose in 41 ms against 12.8 ms at a steady zoom,
  with the same operation counts. Their display copy waits on the GPU (about
  11 ms).
- A drained GPU timeline (`C3X_RENDERER_PROFILE=3`) presented only 25 frames
  in a whole run, too intrusive to attribute these frames.

**Next (4.2).** Make frames during a transition cheap, for example by drawing
them from one completed image of the world rather than re-projecting every
layer at each intermediate scale. That is a visible change, so it needs the
user's decision.

## 43. Stage 4.2: where a zoom frame's time goes (October 9)

**Instruments.**
- Each presented frame's `route-frame-budget` line now also carries:
  - the composition phases (prepare, evaluate, assemble, display);
  - the fused-pass state;
  - cumulative fence, permit and busy counters;
  - `stretch`, the largest magnification of a displayed world image. It
    exceeds 1 when a frame stretches pixels drawn at a smaller scale.
- `C3X_RENDERER_PROFILE=4` drains the GPU only at the phase marks (scene
  prepare, reflection, static, water, units, reconstruct, then the four
  composition phases). Mode 3 also drained at every native operation, which
  left 25 presented frames in a run (section 42).

**CPU (z6, busy save, near scenario).**

| Frames | Compose p50 | Prepare (scene render) | Evaluate | Display | Interval p50 |
| --- | --- | --- | --- | --- | --- |
| Zoom-changing (63) | 47.1 ms | 17.1 ms (12.8 ms) | 3.9 ms | stalls 15–90 ms on every other frame | 54.8 ms |
| Steady (4687) | 15.5 ms | 9.1 ms (8.5 ms) | 1.7 ms | 0.2 ms | 24.6 ms |

- The display stall is the bind of the swap-chain buffer, or its draw.
  - In steady frames it is 60 Hz pacing. It appears only when a frame is
    ready less than ~16 ms after the previous Present, and it waits about
    9 ms.
  - Zoom frames stall even 36–64 ms after the previous Present. They are
    waiting on GPU work.

**GPU, isolated per phase (z7, `PROFILE=4`).**

| Frames | Static | Water | Reflection | Composition | Frame total |
| --- | --- | --- | --- | --- | --- |
| Zoom-changing | 42.7 ms (p90 115.6) | 12.0 ms | 2.7 ms | ~10 ms | 83.7 ms |
| Steady | 0.3 ms | 21.4 ms | 4.9 ms | ~10 ms | 46.7 ms |

(All values are p50 unless marked. Drains inflate absolute values; the
comparison between rows holds.)

**Cause.** Matched with `static-compose`, every expensive zoom frame refines
the static raster toward the destination zoom.
- With refinement: 27–177 ms of GPU per frame.
- Preview only: 1–2 ms.

Refinement's budget follows CPU time (an 8 ms target). Under Parallels its
GPU cost is 8–15 times its CPU time, so each refining frame costs
40–100 ms of GPU.

Effect in z6:
- A notch's first frame starts a refinement and shows 80–140 ms after the
  wheel.
- The transition shows 3–7 frames.
- The destination is sharp 450–525 ms after the wheel.
- The critically damped animation itself reaches its exact value after
  ~350 ms; it is visually settled by ~120–150 ms.

The composition is not the cost: re-projecting the layers adds 2–4 ms.

**Changes tried (same-build A/B, busy save, near scenario, two runs each).**

| Change | Effect | Kept |
| --- | --- | --- |
| Refinement hold: no refinement toward the destination while the zoom is more than 1% away from it (`C3X_RENDERER_ZOOM_REFINE_HOLD`, 0 turns it off) | p90 frame interval 84–102 ms, against 120–138 ms without it. Median interval, first change (49–69 ms) and time to sharp (~500 ms) unchanged. | Yes |
| Shadow-field hold: a moving zoom-in keeps the wider shadow field until it settles; zooming out refits at once | The refit (13–56 ms of CPU) moves from the first frame to the settle point. It also removes the unshadowed edges a field shrunk at once left during zoom-in. | Yes |
| Static budget steered by the presented surface's GPU backlog | No faster frames: interval p50 49 ms against 44–56. Refinement slowed: two zoom-ins were not sharp before the next notch. | No, reverted |

**What limits a zoom frame in normal (undrained) runs.** Drains exaggerate the
static layer: in normal runs its GPU work overlaps the CPU work. A
transition frame at roughly 45–55 ms is made of:
- 10–14 ms of CPU re-rendering the scene at the new scale. Within that,
  unit selection at the new zoom takes ~2 ms.
- 2–7 ms re-projecting the layers.
- Display stalls of 12–28 ms on about every other frame, whether or not
  any static work ran.
- Up to ~26 ms between frames, while Civ III's own interface work holds
  the renderer's lock: the frame's try-lock fails 2–11 times per frame.

There is no single large lever left inside the scene. Smoother transitions
would need:
- display frames that do not wait on Civ III's native submissions;
- or stand-in frames (the opt-in fallback agreed with the user).

**Soft scrolling at 2×/3× (user report).** `stretch` shows that throughout
the 2× and 3× scroll segments the displayed world is Civ III's 1× canvas
magnified, by exactly 2 or 3:
- 43 frames over 4.7 s at 2×;
- 53 frames over 5.3 s at 3×.

During those segments the scene renders at zoom 1.0: the map view takes its
non-projected path. While Civ III scrolls, the view's world holds a
map-derived operation that `projectable()` rejects. The scroll blit (a copy
of the map canvas at the step offset) is the likely one; this is not yet
confirmed. The fix is to keep the view projected during steps, drawing each
step at the presented zoom.

## 44. What holds the renderer during a zoom (October 9)

**Self-inflicted interface work.** Our injected 16 ms view timer
(`custom_renderer_view_timer`), and the same check in the combat-zoom
poll, made Civ III redraw its whole main interface whenever the presented
zoom changed, only to move the minimap box. During an animation the
presented zoom changes on every frame. Zooming out also re-ran a camera
move and a native display update each time.

The box now follows the zoom twice per notch: when it is within 1% of the
target, and when it arrives (`custom_renderer_minimap_zoom_due`; test
`test_camera_navigation`). Zoom-frame interval p50 was 34.6 and 46.6 ms in
two runs, against 44–56 ms before. That is a modest gain, within run noise.

**Who holds the lock.** Route frame budgets now carry `busy_by`: the command
the lock holder waits on when a display frame finds the renderer busy.

| Frames | Busy hits per frame | Main holder |
| --- | --- | --- |
| Zoom-changing | 3.0–3.4 | `gpu_images_scope` (Civ III's native draw batches), 1.9–2.3 per frame |
| Steady | about 0.5 | spread across tactical, step drawing and presentation; native batches about 0 |

During a zoom, frames are not held behind batch receipts (section 42), so
they collide with the batches instead.

**What Civ III draws.**
- **Every tick, even with no input.** The Animator steps every visible
  unit's idle animation:
  - about 49 `C3X_NATIVE_UNIT_DRAW` operations and one full-screen
    hand-off per tick;
  - about 930 compositor commands per tick;
  - about 130 ms of renderer time per 2 s (6.5%), continuously.

  The unit-draw patch forwards animation timing and Civ III's unit
  overlays (flag, health bar, selection cursor). The 3D bodies animate on
  their own clock, so most of this per-tick work redraws unchanged
  overlays.
- **During a zoom.** The map moves under a still cursor, so Civ III's hover
  handler fires about 7 times per notch and re-creates a 36×30 cursor
  surface each time.

## 45. An injected unit-pass skip (tried and reverted); soft unit HUD at zoom (October 9)

**Unit-pass skip.** We tried skipping Civ III's per-tick unit pass when the
Animator's units, selection, camera and zoom were unchanged. On the busy save
it never engaged. Idle ticks still sent every unit draw (16,896 in 20 s),
because the check required every unit to be in its default animation, and
units in other states (for example fortified) failed it. It also reached into
unnamed Animator fields and steered `Animator::update_display` by zeroing its
stored erase rectangle. We reverted it. The user's alternative supersedes it:
the renderer draws the unit HUD, city labels and map messages itself (stage
4.3, in `camera_decoupling_design.md`).

**Soft unit HUD at zoom (user report).**
- At a settled 2× (`stretch` = 1), the selection ring and the unit health bar
  are drawn at twice their 1× thickness and soft. City labels next to them
  stay crisp at their native size.
- While scrolling at 2× and 3×, the whole canvas, labels included, is the
  1× canvas magnified (section 43).

## 46. Stage 4.3, step 1: the held route at display resolution (October 9)

**Problem.** The pathfinder route was drawn by our vector drawer, but
rasterized at 1× into Civ III's unit canvas. That canvas is projected with the
world, so at 2× and 3× the route line, the destination circle and the turn
count were magnified 1× pixels (the user's report).

**Changes.**
- A held route is now a world overlay. `Session::world_overlay` stores it with
  the native canvas it stands for. A native erase over its area retires it, and
  a newer route on that canvas replaces it.
- At each world boundary, the overlay is recorded as a direct operation over
  the world view. That operation samples the presented zoom every frame and
  draws the canonical primitives projected at display resolution, through
  `tactical_world_gpu`, a separate drawer instance that owns whole-view
  textures.
- It sits below the HUD batch, matching its native canvas.
- The tactical anti-aliasing ramp is one display pixel at every zoom. Its
  0.75-canonical floor had widened it to 1.5–2.25 px at 2–3×, which also
  softened the 3D selection ring.
- Recordings carry the overlay flag in the tactical flags word. Older
  recordings still read.

**Evidence.** The `route-zoom` scenario holds a route through 2×, the zoom to
3×, and 3×.
- Before: the line, both rings and the "1" were soft and doubled.
- After: they are crisp at both zooms.

**Tests.**
- `test_retained_composition` ("held route overlay"): drawn at each
  transition frame's own zoom, replaced by a newer route, retired by an erase.
- `test_input_recording`: flag round trip; older and unknown flags.
- `test_tactical_overlay`: GPU coverage.

## 47. Stage 4.3, step 2: the renderer draws the unit status (October 9)

**Change.** Civ III no longer draws a unit's map status: the health bar, the
fortified outline, the movement LED and the stack marks.
- `patch_Unit_draw_map_status` reports, inside its accepted HUD scope, what
  `Unit::draw_status` (0x5BA750) would draw:
  - HP and damage;
  - whether the unit has a bar and is fortified;
  - the stack count;
  - the movement LED, as the JGL sprite inside Civ III's `Sprite`.

  The LEDs are loaded and sliced from `MovementLED.pcx` exactly as Civ III
  does it.
- `C3X_NATIVE_UNIT_STATUS` turns those facts into one command. The renderer
  re-creates the native ink at placement, from `draw_status`'s geometry and
  its 15/16-bit colours, so nothing is drawn into the canvas.
- The renderer may decline any report; Civ III then draws natively, as it
  does with the configuration off.
- `Adapter::ordinary_sprite` decodes plain and row-trimmed 8-bit sprites.
  The LED slices are row-trimmed.

**Evidence** (busy 1498 AD save, `near` scenario):
- Trace: 122,880 reports accepted, none refused; no visual failures.
- 1×: the bars, LED and stack marks at Nagoya and Kagoshima match the
  native capture within JPEG noise. The largest channel difference over the
  bar regions is 26.
- 2×: the same native-size status beside the crisp ring.
- Idle native-image execution (per 2 s):

  | | Commands | Renderer time |
  | --- | --- | --- |
  | Before (section 44) | about 24–33k (about 930 per tick) | about 130 ms |
  | After | 6.2k (about 240 per tick) | about 50 ms |

**Tests.**
- `test_unit_status_report`: draw_status's rules become facts, and the JGL
  sprite is what gets passed. It fails when the `Sprite` wrapper is passed.
- `test_ordinary_sprite_decode`: plain and row-trimmed decode, plus
  refusals. It fails without trimmed support.
- `test_retained_composition` ("renderer unit status"): ink, placement and
  erase.
- `test_unit_hud`, `test_custom_zoom`, `test_injected_unit_bootstrap`: the
  report is offered inside the scope; native draws when the renderer
  declines and when the configuration is off.
- Harnesses broken by sections 44 and 46 are repaired:
  - `test_native_view_identity`: the minimap follow rule is excluded; it
    has its own test.
  - `test_frame_publication`: world overlay stubs.

**Not reproduced.** Mech Infantry's 4 px quirk depends on `FUN_00539290`,
which is unavailable. The stack count uses `patch_Unit_is_visible_to_civ`,
because the exact native check, `is_unit_hidden_from_player`, has no usable
`civ_prog_objects.csv` entry.

## 48. Combat: the display fell seconds behind Civ III (October 9)

**Found while checking the unit status in combat** (`combat` scenario,
`user-3350BC.SAV`):
- Bombard: the defender's bar went from green 3 to yellow 2 to red 1, as it
  should, but 1.6–1.9 s after Civ III applied each hit. The 3D impacts reached
  the screen 0.2–0.35 s after theirs.
- Melee, three runs: each failed with `async-publication-failed reason=renderer
  publication pressure` at the 65,536 work limit. After that, the GPU image
  session refused every operation for the rest of the run (about 78,000
  failures).
- It is the same with Civ III drawing the statuses (the new
  `C3X_RENDERER_NATIVE_MAP_HUD=1`), so section 47 did not cause it. The earlier
  clean combat captures (bombard, victory, air) were single-round; a long
  melee had not been validated.

**Cause.**
- During a fight, Civ III's loop ticks the animating units about 270 times a
  second: 1,600 unit ticks/s against 78/s at idle. It presents about 60 times
  a second.
- On the bridge's transport thread, each present costs about 5.5 ms and each
  image batch about 6 ms. Between 43 and 48 s they took about 4.4 s of every
  5 s.
- The publication queue grew steadily. A profiled run reached 4.5 s of latency
  and 59,500 work units.
- Image batches wait for capacity and filled it to the limit. Ordinary facts,
  such as unit observations, never wait, so the next one was rejected and the
  session faulted.
- HUD ink and statuses travel with the image batches, so they were as late as
  the queue. Combat-effect facts may pass queued canvas work, which is why the
  impacts were on time.

**Changes.**
- At most two native presents in flight (`AsyncSceneClient::present`).
  - A third waits on queue progress (`Publication::wait_progress`). This
    paces Civ III's loop to the renderer, like a maximum frame latency.
  - A token captured in the present's work counts it retired on every path
    (executed, abandoned or rejected), so a fault releases the wait.
- Waiting producers leave an eighth of every budget (bytes, packets, work)
  for ordinary state publication. An empty queue still admits any request
  that fits.
- `C3X_RENDERER_NATIVE_MAP_HUD=1` declines the renderer-drawn HUD, so Civ III
  draws it. This is the side-by-side reference for stage 4.3.

**Evidence** (same save and cases):

| | Before | After |
| --- | --- | --- |
| Melee publication failures | 3 of 3 runs | none (2 runs) |
| Peak queue latency during the fight | 0.7–4.5 s | 40–93 ms |
| Peak queued work | 59,500 units | about 200–430 units |
| Bar change after each bombard hit | 1.9 s, 1.6 s | 0.22 s, 0.16 s |

In the melee, the attacker's bar shows its damage. When the attacker dies,
its bar and stack marks disappear and the next unit on its tile shows its
own bar. The defender ends at yellow 2 of 3.

**Tests.**
- `test_present_frames_in_flight`: a third present waits for the first, and
  a fault releases it. Fails without the wait.
- `test_async_image_backpressure` ("waiting images leave room for ordinary
  facts"): fails on the old admission. The full-window upload case now
  expects ten 11.3 MB windows below the 7/8 watermark, where it expected
  eleven.

## 49. Stage 4.3, step 3: city labels, measured; native strokes become fills (October 9)

**What labels cost** (busy save, `near`, 88–150 s; new `hud_city` count in
`native-image-execution`):
- City-label HUD items received 19–26k commands: about 12–18% of native-image
  commands while the camera moves, and none at idle.
- The bridge's transport thread spent about 29 s of the 62 s:
  - image batches: 1,702 posts at 8.6 ms each;
  - presents: 779 posts at 9.2 ms each;
  - state facts: 18,887 posts at 0.12 ms each;
  - camera adoption: 56 posts at 35 ms each;
  - tactical records: 889 posts at 2.1 ms each.
- The cost per batch or present is mostly waiting for the renderer, not
  label content.

**Native strokes become fills.**
- Civ III draws each city label's border as four one-pixel OpenGL lines,
  which C3X turns into tactical captures.
- The tactical native line, with +.5 centres and hard caps, covers exactly the
  half-open run of pixels from its start toward its end. The GPU test checks
  this in all four directions, black and white.
- The owner now fills such strokes (solid, opaque, axis-aligned, black or
  white) and skips the tactical raster.
- Result:
  - tactical records while the camera moves: 889 → 0;
  - image-batch posts: 1,702 → 1,002, because strokes no longer split
    batches.
- Total transport time is unchanged within noise (29.8 → 29.0 s), because it
  is mostly waiting.

**A facts-based city label (not done; for the user's decision).** Civ III's
`draw_city_hud` is now fully specified (geometry, colours, strings, fonts),
and the bridge can rebuild labels from facts with Civ III's own fonts.
- Font: `font+4 → +0x18` is the HFONT. Lucida Sans, lfHeight −size, weight
  0 or 700, OUT_TT_ONLY_PRECIS, DEFAULT_QUALITY.
- Text placement: TA_BASELINE at `top + (h − m)/2 + m`, where
  m = tmAscent − tmInternalLeading.

Replacing the native pass, though, needs `civ_prog_objects.csv` entries that
do not exist:
- the harbour, airport and veteran sprite globals (GOG 0xB3F380, 0xB3F354,
  0xB3F328);
- `City::spawns_veteran_ground_units`;
- the advisor-form status check (0x9AFD98);
- `Leader::is_tile_explored`.

It also needs a large port into injected code. Because the bridge already
translates each native label op (Civ III's own pixel work is skipped), moving
the same work into the bridge saves little. The real win is labels as world
state:
- keep an unchanged label instead of re-recording it;
- re-anchor it on camera moves instead of redrawing.

**Soft zoomed scrolling: cause found** (`C3X_RENDERER_ROUTE_WITNESS=1`):
- While scrolling at 2×, about 5.6 frames are presented per second, and 17 of
  28 are stretched ×2. At 3×, 23 of 30 are stretched ×3.
- The world view stays projectable: the new `world-view-projection` trace
  never fired. The section 43 hypothesis is wrong.
- Every camera step's scene renders at 1.0 (lane 0, mostly previews). The
  ready-frame job, which renders at the presented zoom, returns while a camera
  job is active or after a newer map publication (`gpu_serial` changed).
- With a step every 66 ms and each 1× camera job longer than that, the zoomed
  render never runs until the camera stops.

**Tests.**
- `test_native_stroke_fill`: runs, reversals and refusals. Fails without the
  change.
- `test_tactical_overlay` (GPU): exact stroke runs.

## 50. Scrolling zoomed in draws each step at the presented zoom (October 9)

**Problem** (section 49): while scrolling at 2× and 3×, 60–75% of presented
frames were the step's canonical (1×) image magnified, because the zoomed
render waited for the camera to stop.

**Changes** (`c3x_renderer.cpp`):
- When Civ III adopts a step and the zoom is settled at a value other than 1,
  the step is also prepared at that zoom right after its publication. The
  canonical import is untouched.
- The step's projected sampler returns that completed zoomed draw even while
  the next camera job runs. Holding instead showed the canonical image
  magnified.
- A blocked `prepare` no longer clears the job's readiness. The compositor
  calls it every display frame; clearing first discarded the zoomed draw.

**Evidence** (busy save, `near`, `C3X_RENDERER_ROUTE_WITNESS=1`; new
`Renderer/tools/zoom_sharpness_report.py`):

| Zoomed scroll seconds | Soft frames before | Soft frames after |
| --- | --- | --- |
| 2× (42–46 s) | 17 of 28 | 1 of ~30 |
| 3× (52–56 s) | 23 of 30 | 0 of ~35 |

- Steady 2× and 3× stay at 59–61 fps. One run read 24–43 fps; a repeat
  showed that was VM noise.
- Scroll frame rate is unchanged and still low: about 4–11 fps at every zoom
  on the busy save in the VM. That is the next problem: 3,000 composition
  operations a frame, lock waits behind camera jobs and image batches.

**Tests.** `test_zoomed_step_sample`:
- the sampler returns the completed zoomed draw during a camera job, and
  fails without the change;
- a blocked `prepare` keeps readiness.

## 51. The world window on by default; deferred steps drawn at the presented zoom (October 10)

**Finding.** Stage 2's resident world window (2a) and its deferred scroll
steps (2c) worked only with `C3X_RENDERER_WORLD_WINDOW=1`. Ordinary games and
every capture since section 41 refused all deferrals (`deferred-step-refused
reason=1`). Stage 2 finished in section 41, so the window is now on by
default, and `C3X_RENDERER_WORLD_WINDOW=0` opts out.
- The renderer's control and the capture margin it returns to Civ III follow
  the same default.
- Deferred steps are drawn canonically at adoption. Zoomed in, they now also
  get the section 50 presented-zoom draw. Without it, 26 of 250 (2×) and 25
  of 571 (3×) frames were magnified.

**Evidence** (busy save, `near`; mean fps over the seconds of each scroll
segment):

| Run | 1× scroll | Zoomed scroll | Soft zoomed frames | Failures |
| --- | --- | --- | --- | --- |
| Window off (h14) | 6.7 | 6.7 | 5 | 0 |
| Window on, before the deferred-step fix (w20) | 14.2 | about 8 | 51 | 0 |
| Window on (w21) | 10.0 | 6.4 | 2 | 0 |
| Default, no option (w22) | 11.2 | 7.7 | 1 | 0 |

- Melee combat with the default: finished, with no failures.
- Steady 2× and 3× stay at 55–61 fps.

**Where a scroll step's time goes now** (window off, 1×):
- The camera job takes about 100 ms from render-begin to camera-complete.
- The next job starts 200–270 ms later: adoption, Civ III's redraw of the
  step, and its image batches, each waiting about 9 ms for the helper's
  single worker.
- Civ III's own thread barely waits (4–45 ms per 2 s).
- The remaining lever is the single worker that runs camera jobs,
  composition and native batches in turn. That is the planned job split.

**The busy 1× HUD** (new `hud_items_max` and `hud_placed_max` in
`native-image-execution`):
- About 150 HUD items (about 50 city labels and 100 unit statuses) place
  about 3,100 draws. At 2× and 3× it is 12–34 items and 200–750 draws.
- At idle 1×, the fused pass replays all 3,080 every frame, because the
  animating scene below changes. That costs little CPU: evaluate is 1.7 ms of
  a 22.7 ms frame, and the scene prepare is 14.6 ms.
- Zoomed scroll is just as slow with five times fewer HUD draws, so HUD
  replay is not what limits scrolling.

**Tests.**
- `test_world_window_default`: unset or `1` is on, `0` is off, for both the
  control and the margin. Fails on the old opt-in parse.
- `test_zoomed_step_sample`: the adoption prepares every step, deferred ones
  included, after the canonical publish.

**Amendment (same night): the window is opt-in again.**
- On the light save (`near`) it ran camera jobs continuously at steady zoom:
  49 completions in 4 s at 2×, so every zoomed frame was magnified (100%
  soft), at about 13 fps, and the final 1× segment ran at 12.9 fps.
- With `C3X_RENDERER_WORLD_WINDOW=0`, the same save ran at 60 fps at steady 3×
  and in the final 1× segment.
- The window is opt-in until that is fixed. Section 41 had validated it only
  on the busy save.
- The adoption-time zoomed draw now runs only when the step moved the camera.
  A same-view republication keeps its completed scene, and drawing one per
  republication slowed the light save (1× segment 38 → 48.6 fps).

Defaults now (window off):

| Save | Steady 2×/3× | Zoomed scroll | Soft zoomed frames | Failures |
| --- | --- | --- | --- | --- |
| Busy (b3) | 58–62 fps | 4–11 frames/s | 2 | 0 |
| Light (l3) | 60 fps (3×) | 13–22 frames/s, about the step pace | 34 at 2× | 0 |

During a zoomed scroll a frame is presented when its content changes, so
frames per second there is close to the step rate. The October 7–8 light-save
scroll figures (42–46 fps) were measured with the glide that was later
removed.

**Why the window failed on the light save** (l4, window on). The run never
prepared a map frame (`frame-preparation-ready`), so the presented-zoom path
never engaged:
- every zoomed frame was the canonical image magnified;
- idle animation dropped to about 13 presents a second;
- the helper's camera events stop at 62.5 s.

Cause not yet known.
- On this small map the whole world fits the window (box 0,0,4096,2304).
- Candidates: `retain_visual_map` returning no prepared sampler, and the
  camera state (`camera_active`, `camera_scene_complete`) staying gated after
  the last deferred step.
- `map-path fresh` only names the scene path, not the frame cache path that
  `gpu_publication.fresh` checks.
- A temporary trace (l5) showed the actual failure. With the window on,
  Civ III adopted only the first step: one `gpu-map-publication` against 211
  `camera-complete` (mostly deferred), and `gpu_serial` stayed at 1.
  - Every later frame reprojected that stale first map: soft when zoomed,
    with dead idle animation.
  - The save is an early game whose explored world fits inside the window.
- Fix that adoption stall, and validate on the light save, before enabling
  the window again.
