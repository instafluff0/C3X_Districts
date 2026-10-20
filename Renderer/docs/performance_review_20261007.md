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
- Where the stall comes from (temporary traces, l6–l8, light save, window on):
  - Bridge map transactions: 975 prepares and 211 polls returned
    `SUPERSEDED`, each followed by a new begin; there was one adoption.
  - The `SUPERSEDED` comes from the async client's completion check
    (`same_camera_source` in `async_scene_client.h`). The helper's completed
    camera frame and the bridge's latest published frame are identical
    except one tile, index 1722 of 3,340.
  - On that tile the flags differ: `0x12040` (KNOWN|PREFETCH|TOPOLOGY_HALO,
    the world window's rewrite in `world_window.h`) against `0x12001`
    (KNOWN|PREFETCH|RENDER).
  - The tile's map position changes from run to run (44,4; 0,4; 10,4…).
    Either the window's halo rewrite reaches the echoed frame, or
    consecutive captures legitimately differ while scrolling.
  - Even so, adoption never resumes after the camera stops.
- Fix that adoption stall, and validate on the light save, before enabling
  the window again.

## 52. The light-save stall: wrapped copies in the camera delta (October 10)

**Cause.** The `0x12040` flags came from the injected capture margin, not
from `world_window.h`: tiles past the zoom envelope are captured as
`TOPOLOGY_HALO | PREFETCH` (section 30). On a small wrapping map, the
window's wider margin (`WorldWindow::margin_x/margin_y`) reaches around the
world, so one tile is captured twice: on screen as RENDER and as a wrapped
copy as HALO.
- The camera delta (`camera_delta.h`, section 30) keeps one slot per
  canonical coordinate on the helper and overwrites it while decoding.
- The sender judged each occurrence against the previous frame, so the
  later copy was sent as "unchanged" and rebuilt with the earlier copy's
  flags.
- The helper's frame then differed from the bridge's request on every step.
  Every completion was superseded, and Civ III adopted only the first.
- Large maps have no wrapped copies in the margin, so the busy save never
  showed it.

**Fix.** The sender follows the receiver's slot within the frame
(`CameraDeltaSender::written`). Wrapped copies with different flags are
sent in full each step; unique tiles still travel as references. Test:
`test_camera_delta.py`
(`test_wrapped_copies_with_different_placement_rebuild_exactly`). It fails
on the old sender with the flags seen in the game (`12040` against `12001`).

**Result** (`near`, window on, same build; soft = drawn below the presented
zoom):

| Save | Adopted / completed | 1× | 2× | 3× | Final 1× | Failures |
| --- | --- | --- | --- | --- | --- | --- |
| Light, window off (l3) | — | 48.6 fps | 15.6 fps, 22% soft | 36.3 fps | 60.0 fps | 0 |
| Light, window on (l9) | 154 / 154 | 51.7 fps | 18.7 fps, 2% soft | 44.6 fps | 60.0 fps | 0 |
| User, window on (u2) | 191 / 191 | 49.1 fps | 27.1 fps, 11% soft | 36.1 fps | 59.7 fps | 0 |
| Busy, window on (b5, b7) | 68 / 68, 70 / 70 | 26.8, 32.0 fps | 21.7, 23.4 fps | 36.7, 37.1 fps | 33.6, 32.9 fps | 0 |

Before the fix the light save adopted 1 of 211 completions.

**The idle cost was the witness.** A same-build busy A/B with
`C3X_RENDERER_ROUTE_WITNESS=1` showed the window 10–18% slower at idle
(35–36 against 38–41 fps; frame composition 13.3 against 10.3–11.1 ms). The
witness hashed every captured tile byte on every frame, and with the window's
capture margin the frame carries 4,888 tiles (2,304 of them window tiles).
Without the witness, idle frames per second (from `frame-preparation-ready`):

| Busy save | Opening 1× idle | Final 1× idle |
| --- | --- | --- |
| Window on (b9, b11) | 43.1, 43.3 | 39.1, 34.3 |
| Window off (b10, b12) | 33.9 (a noisy run), 45.4 | 38.4, 38.7 |

- The witness now hashes a publication's tiles once
  (`RendererState::route_source`; digests unchanged). Test:
  `test_route_source_digest.py`, which fails on the per-frame hash.
- Comparisons between capture sizes made with the witness before this change
  (sections 49–51) understate the larger capture by 1–3 ms a frame.
- Zoomed and 1× scroll rates on the busy save are about the same with and
  without the window (5–6 frames a second at 1×).
- **The window is on by default again** (`C3X_RENDERER_WORLD_WINDOW=0` opts
  out): neutral on the busy save, faster and sharper on the light save, and
  the user save adopts every step.

**Default build** (no window option, d10 light, d11 busy): the window was
valid on every step (190 and 69), every completion was adopted, and no run
failed. Light: 52.4 fps at 1×, 23.4 at 2× (0% soft), 44.7 at 3×, 60 at the
final 1×. Busy: final 1× idle 38.3 fps with the witness on (now cheap).
The busy save's first seconds at 2× ran at 4–16 frames a second in this run;
watch it in later runs.

## 53. Where a busy scroll step's time goes with the window (October 10)

Busy save, `near`, window on, trace level 2, no witness (b9, b11). Step
period is adoption to adoption.

| Step | Steps | Median period | Scroll time |
| --- | --- | --- | --- |
| Full job (mostly block crossings) | 19, 20 | 427, 426 ms | 8.4, 8.2 s |
| Covered (deferred, drawn at adoption) | 48, 51 | 212, 191 ms | 11.4, 10.5 s |

**A crossing job (330–810 ms):**
- 110–350 ms building the entering band's tiles on the worker
  (`render()` p5, the per-tile loop: routes, features, cliffs, resource
  anchors, `cache_geometry_layer`, uploads);
- 36–173 ms of display frames run inline at the job's checkpoints
  (`service_camera_preparation`; about 70 ms each);
- 9–175 ms waiting for the band's RAM-backing restore, which runs on
  `world_preparation_queue` workers but is only scheduled inside p5;
- 75–210 ms of draw (`sandbox_fresh.draw`).

**A covered step** uses about 12–25 ms of setup and 20–60 ms of draw on the
worker; the rest of its period is Civ III's ticks.

Notes on the traces: `frame geometry_ms` is the whole setup (p1–p7), not part
of the draw; `draw_submit_ms` is the draw. Setup phase times include display
frames serviced inline; subtract `camera-service-turn` before attributing.

**Consequence for the order.** Whole-world residency builds each tile once
instead of at every crossing (43% of scroll time). The worker split could
move only about 15 ms of CPU setup per covered step off the context thread,
and display frames swap the completed scene into the camera job's live
fields (`borrow_completed_scene`), so a concurrent job needs separate state
first. The user agreed to do whole-world residency first.

## 54. Whole-world residency, step 1: one appearance per explored tile (October 10)

**Why crossings rebuilt tiles.** Loading makes the whole explored world
resident (`world-gpu-residency fully_resident=1`, 16,900 keys, 1.98 GB), but
the map draws Civ III's per-step capture. For explored fog the world seed
sent a narrower record (no resources, tile buildings, effects or border
edges), so its world input was not "full" and the loaded content was a
different appearance from the one the window drew. Each crossing into fogged
land restored or rebuilt those tiles; each new variant evicted others, and
evictions of shared natural components invalidated more (`shared`). Native
draws those facts on fog too: `Map_Renderer::m19` passes the same flags for
every tile and `m33_Draw_Resource` checks only `get_resource_visible_to`.
The geometry budget in the VM is about what is owned plus 0.1-0.3 GB (free
RAM 3.7-4.0 GB against a 3.4 GB reserve), so every new variant evicts.

**Change.** The seed reads explored fog with the capture's own
`read_custom_renderer_tile` (no units) and marks it `HALO | PREFETCH`, as it
already did for visible tiles; unexplored tiles keep the topology record
(`injected_code.c`, ledger "Explored fog seeded by the capture's read").
Test: `test_injected_world_authority.py` (five cases fail on the old seed).
Explored resources also keep animating in fog, as water does (the user):
`test_fogged_resource_motion.py`.

**Result** (busy, `near`, window on, trace level 2, no witness):

| Crossings | Before (b9, b11) | After (f1, f2) |
| --- | --- | --- |
| Tiles built | 2,445 | 179 |
| Tiles evicted | 3,876-3,877 | 569-616 |
| Missing bindings | 1,439 | 0 |
| Shared-content rebuilds | 2,012 | 358 |
| Crossing job p50 / p90 | 211-227 / 644-749 ms | 149-164 / 210-369 ms |
| Full step period p50 | 421-427 ms | 298-319 ms |

Covered steps are unchanged (p50 166-211 ms). f2 completed 83 scroll steps
against 67-71 before in the same scroll time. Idle is unchanged (busy final
1× 39.4 and 37.2 fps; light 60). No failures; every completion adopted.

**Left:** 358 shared-content rebuilds and about 600 evictions per run; a
crossing still costs about 100 ms more than a covered step (membership,
shadow proofs and static repair for the entering band, review 53).

## 55. Shared components outlive their dependents (October 10)

**Cause of the shared rebuilds.** Eviction ranked a region's shared natural
component by its own last use. Everything loaded at start shares old use
stamps, so a component could go before its tiles; every resident tile drawing
with it was then invalid (`tile-dependency-reasons shared`) and rebuilt when
the window reached it, while still occupying memory.

**Change.** `ResidencyCandidates` ranks a component as recently used as its
newest resident dependent and, on a tie, after it; a dependent in the current
frame pins it. Test: `test_residency_candidates.py`
(`test_shared_components_outlive_their_resident_dependents`).

**Result** (busy, `near`, e1 and e2 against f2):

| | f2 | e1, e2 |
| --- | --- | --- |
| Shared-content rebuilds | 358 | 0 |
| Tiles built at crossings | 179 | 96 (evicted tiles only) |
| Evictions | 616 | 603, 636 |
| Crossing job p90 | 210 ms | 339, 181 ms |
| Full step time | 7.9 s | 6.6, 7.1 s |

Light save: 196 of 196 adopted, no failures. The remaining evictions come
from the VM's geometry budget, which has no growth margin (section 54);
stage 5's smaller world is what removes them.

## 56. How a covered step scales with the window (October 10)

Temporary experiment (reverted): the window's reach ×3 on the busy save
(x3), against e2. With explored fog seeded in full (section 54), the
window drew 1,344 tiles beyond Civ III's capture from world inputs, with no
skips, no failures and every completion adopted.

| | e2 (reach ×1) | x3 (reach ×3) |
| --- | --- | --- |
| Window tiles | 2,304 | 5,472 |
| Covered render setup | 11.7 ms | 17.8 ms |
| …membership admission (p4) | 2.6 ms | 6.5 ms |
| …window build (p1) | 1.6 ms | 2.3 ms |
| Covered render draw | 23.0 ms | 35.0 ms |
| …traced scene phases within it | 10.8 ms | 11.9 ms |
| Crossings | 26 | 22 |
| Crossing builds / evictions | 96 / 636 | 642 / 2,363 |

A covered step costs about 5.7 µs more per resident tile. At the whole busy
world (16,900 tiles) that is about 80 ms more per step. Whole-world residency
therefore needs per-step work bounded by the view, not by the resident set:
membership admission, the window build and the draw's per-record work
(most of its growth is outside the traced scene phases; attribute it first).
A larger window also evicts more while the VM's geometry budget has no
margin.

**Attribution** (reach ×3 rerun x4 against e2; covered camera draws only,
pairing each `render-setup-phases` with the traces written before its
`frame` line):

| | ×1 | ×3 |
| --- | --- | --- |
| Setup | 12.1 ms | 19.5 ms |
| …membership admission (p4) | 2.6 ms | 7.4 ms |
| Draw | 23.1 ms | 37.1 ms |
| …water | 2.2 ms (120 draws) | 5.4 ms (714 draws) |
| …static (refinement appears) | 5.6 ms | 8.3 ms |
| …shadow caster collection | 1.8 ms | 3.9 ms |
| …reflection | 0 ms | 1.9 ms |
| …units, prepare | 3.0, 2.8 ms | 4.0, 3.7 ms |

The view is unchanged, so six times the water draws means water is culled
against something that grows with the window. Order of work: water culling,
membership admission, shadow caster collection, static refinement.

## 57. Civ III's native capture envelope again (October 10)

The window's capture margin (sections 29-31) existed because window tiles
past Civ III's capture had no full appearance. Since the seed reads explored
fog as the capture does (section 54), every explored tile's world input is
complete, so the window can draw those tiles from the retained world. Busy
save, window on, margin 0 (m1) against the margin (e2):

| | e2 (margin) | m1 (margin 0) |
| --- | --- | --- |
| Native capture | 4,888 tiles | 1,686 tiles |
| Window tiles from world inputs | 0 | 1,224 (none skipped) |
| Covered render setup / draw | 11.7 / 23.0 ms | 10.7 / 18.7 ms |
| Covered step period p50 | 174 ms | 156 ms (two ticks) |
| Steps in the scroll segments | 82 | 89 |
| Crossing builds | 96 | 96 |

No failures; every completion adopted. Civ III's per-step capture (three
copies, a comparison and an encode per tile on its thread, section 29) is
about a third of its size. `C3X_NATIVE_CAPTURE_MARGIN` now returns 0.
Tests: `test_world_window_default.py`.

## 58. A still zoomed view kept its magnified image (October 10)

**Symptom.** On the light save, the last two seconds of the 2× scroll segment
were magnified 1× frames (all frames soft), in window-off runs too (l3, 22%
soft). With the capture margin removed, steps came every tick and the
pattern repeated on every run.

**Cause** (temporary traces of the sampler and of prepare's exits). The last
step's adoption prepared its 2× draw, but the next display frame's prepare
cleared `job->ready` at its top and then returned early: an unchanged, still,
unanimated view returns before drawing on every frame. Readiness never came
back, so every frame sampled the canonical image magnified until the zoom
changed. Busy saves keep animating, so they always re-prepared.

**Fix.** Prepare no longer clears readiness; it changes only with the
completed swap (the displayed front is never written meanwhile). The
adoption's initial image now requires a 1× draw (`prepared_map->zoom==1.f`).
Test: `test_zoomed_step_sample.py`
(`test_no_early_return_discards_the_completed_draw`, fails on the old code).

**Result** (r1 light, r2 busy, margin 0 with the fix): light 2× and 3× 0%
soft (19-20% at 2× before), 210 of 210 adopted; busy 2% soft at 2× and 3×,
80 of 80 adopted; no failures.

## 59. The window cannot grow to the whole world inside the frame (October 10)

Temporary experiments at margin 0 (reverted):

| | r2 (×1) | x5 (×3) | x6 (×5) |
| --- | --- | --- | --- |
| Window tiles | 2,304 | 5,472 | 8,320 |
| Covered setup | 10.9 ms | 16.2 ms | first camera job failed |
| …membership admission (p4) | 3.1 ms | 6.4 ms | |
| …window build (p1) | 1.5 ms | 2.4 ms | |
| Covered draw growth | | about +2 ms | |

At margin 0 a covered step grows about 1.8 µs per window tile in setup and
under 1 µs in the draw. At ×5 the first camera job failed (`gpu-failure`,
render returned false): the window is substituted for Civ III's capture as the
camera job's frame, and a frame holds at most 8,192 occurrences
(`CapturedScene::occurrence_limit` and the same cap in visibility coverage,
dynamic scene input, city overlays, unit arrivals, unit instances, the
membership diff and publication). The busy world has 16,900 explored tiles.

A bigger window also does not remove crossings: its box still moves a block
at a time with the camera. Only a fixed whole-world set does, and that set
cannot be a frame. Whole-world residency therefore needs the resident world
separated from the per-step frame: the frame describes the view (Civ III's
capture), and the renderer keeps the explored world resident and consults it
by block for selection, shadows, the static layer and visibility. Raising every
8,192 cap instead would double each per-frame pass and buffer.

## 60. Resident world restructure, R1: view consumers read the native frame (October 10)

The camera job passed the world window (resident membership) to arrival
visibility, visibility coverage, wave chunks, unit selection and the fresh
draw; display frames passed Civ III's capture. Coverage therefore rebuilt
after every step and each consumer paid for the whole window. They now take
the native frame (`render()`; `fill_output`'s ownership mapping is unchanged).
Test: `test_view_consumers_native_frame.py`.

| | r2 | v1 (R1) |
| --- | --- | --- |
| Coverage builds in the run (busy) | 135 | 92 |
| Covered camera draw p50 (busy) | 32.8 ms (noisy run) | 23.1 ms |
| Full step period p50 | 384 ms | 302 ms |
| Steps in the scroll segments | 72 | 84 |

No failures or draw failures on the busy, light (211 of 211 adopted, 0%
soft) and user (205 of 205, 0% soft) saves.

## 61. Why a busy covered step takes two ticks (October 10)

**Civ III's cadence.** Civ III alternates ticks while scrolling: one tick
moves the camera and requests the step, the next re-requests and polls; it
moves again only after adoption. A step takes one tick only if its completion
is ready within the tick it was requested (the light save does this). On the
busy save a covered step completes 63 ms (p50) after its first request.

| Busy covered step (v1) | p50 |
| --- | --- |
| completion → adoption draw starts | 63.5 ms |
| adoption draw → publication | 41.9 ms |
| publication → next completion | 62.4 ms |

**Tried and reverted: camera first.** Starting a pending camera job ahead of
a waiting independent command on the helper's worker (`C3X_RENDERER_CAMERA_FIRST`)
did not speed completions (62.6 → 74.8 ms) and slowed the busy save (86 → 68
adoptions; covered steps 177 → 227 ms; 2× 26.7 → 15.3 fps).

**Where the request waits: the bridge transport** (`publication-latency`,
cf0, busy):

| Operation | Queue p50 / p90 | Service p50 / p90 |
| --- | --- | --- |
| camera-begin | 29.4 / 95.6 ms | 15.5 / 98.1 ms |
| camera-adopt | 1.3 / 139 ms | 53.3 / 116.8 ms |
| images | 0.1 / 105.7 ms | 13.0 / 31.7 ms |
| present | 17.2 / 81.1 ms | 0.3 / 20.7 ms |

One transport thread carries every command synchronously, so a camera request
queues behind the adoption draw (53 ms, synchronous) and Civ III's image
batches, then its own call can block on the helper. The next lever for one
tick per covered step is the transport: camera requests and polls on their
own path, and the helper's camera-begin not waiting on the draw. About 26 ms
of image-batch worker time and 42 ms of adoption draw per step fit one tick.

## 62. A bounded camera poll wait does not give one tick per step (October 10)

Tried and reverted: Civ III's camera poll waited up to N ms for the
completion mailbox (busy, `near`, same build):

| Poll wait | Adoptions | Covered step p50 | Full step p50 |
| --- | --- | --- | --- |
| 0 ms (pw0) | 77 | 227 ms | 338 ms |
| 30 ms (pw30) | 74 | 223 ms | 419 ms |
| 60 ms (pw60) | 69 | 220 ms | 416 ms |

Waiting blocks Civ III's thread without completing steps inside the tick: a
covered step's serial chain on the helper (adoption draw about 42 ms, Civ III's
image batches about 26 ms of service, the request's queue and lock waits)
exceeds one 78 ms tick in this VM, and `begin(n+1)` cannot precede `adopt(n)`
(it clears the ready camera). One tick per busy step needs less work per step,
not reordering.

The image batches' own execution is small (`native-image-execution`
total_ms 40-200 per 2 s window, mostly city labels: up to 6,081 HUD commands
per window); most of their service time is waiting for `call_mutex`, the
worker and the round trip.

This VM is Windows on ARM: Civ III (x86) and Renderer64 (x64) both run under
emulation, so every CPU cost here is inflated. A native x64 measurement of the
busy save would show how far the per-step chain is from one tick on real
hardware.

## 63. Stage 5 census: resident geometry by layer (October 10)

New `geometry-census` trace (with `C3X_RENDERER_MEMORY_CENSUS=1`, every 5 s):
per geometry layer, unique GPU buffer bytes, logical bytes (vertices × stride
+ indices), chunks, vertex and index counts, instances and vertex strides.
One vertex buffer serves several layers of a tile, so unique bytes fall on the
first layer counted; logical bytes are the per-layer split.

Busy save during play (mc1): 4,608 resident owners (2,304 shared natural),
813 MB unique, 945 MB logical (the whole explored world loaded at start is
about 2 GB). Logical MB by layer:

| Layers | MB | Vertex stride | Note |
| --- | --- | --- | --- |
| natural terrain, mountain, decal | 207, 155, 93 | 88 B | 26 M indices, 32-bit |
| underlay, river, water, bed, route, land | 108, 62, 43, 43, 39, 1 | 168 B | water and bed are the same mesh |
| city | 99 | 88 B | |
| farm, feature, mine, site, cliffs | 64, 21, 6, 1, about 1 | 32-48 B | instanced parts |

Light save: 39 owners, 3 MB; its memory is fixed costs.

**Stage 5 direction.** Vertex formats dominate. The 168-byte integrated
vertex (`integrated_terrain.hlsl`) stores per-tile constants on every vertex
(base and real terrain, surface kind, effect, tundra) and full floats for
weights, normals and UVs; a packed layout of about 40-48 bytes would cut those
layers by about 70%. The 88-byte natural and city vertices and 32-bit indices
(16-bit where a chunk has under 65,536 vertices) are the next most valuable.
Changes need Lab comparisons for fidelity.

**Corrections (vertex format map, October 10).** Natural vertices are 92 B
(the census strides mask buckets stride/8, so 88 and 92 coincide); indices are
already 16-bit where a chunk allows; water and bed alias the underlay buffers
with `byte_count=0`, so logical 945 MB double-counts about 86 MB and 813 MB
unique is the resident figure. Vertex bytes: natural 425 MB, city 98.8 MB,
168-B ground layers 208 MB. Plan in the design doc (stage 5).


## 64. Stage 5, step 1: exact compact natural terrain vertices (October 10)

**Change.** A natural mesh drawn only through world reprojection (world
objects on; `natural_projection.z>0` in every natural vertex shader) whose uv
equals world xy and whose relief owner is zero, bit for bit on every vertex,
now packs 56 bytes instead of 92: world xyz+valid, normal, four material
weights, relief height/blend, base terrain
(`render_core/prepared_mesh.h`, `natural_compact_stride`). Its input layouts
read position and uv from the world bytes and the owner from a 16-byte
stride-0 zero stream: the terrain, mountain and object natural layouts
(`source_fidelity/runtime.h`, chosen per chunk by `bind_format` in both draw
loops), both shadow caster layouts (`source_shadow.h`, `fresh_pipeline.h`) and
territory borders. No shader changed. Any vertex outside the rule keeps the
mesh at 92 bytes: mountains (volcano owner) and hill decals (rotated uv) stay
92. The CPU record tier, which stores 23 floats, never compacts and refuses a
compact mesh; terrain compile keys separate the two formats.
`C3X_RENDERER_COMPACT_VERTEX_CONTROL=1` keeps every natural vertex at 92.

**Verification.**
- `test_natural_compact_vertices.py` packs the same vertices both ways and
  fetches every shader input through the layout offsets read from the source
  (main, caster, border): identical bits. A wrong offset fails it.
- Lab A/B, compact against the control, byte-identical BMPs in every pair that
  rendered on both sides: grassland (3 cases), hills (2), mountains (3),
  borders crimson-fine at 64 and 128 and crimson-brush at 64, transitions
  gameplay, volcanoes detail at 128 and 224, floodplains detail (16 pairs).
  The grassland detail scene's geometry: 28.6 → 23.4 MB.
- The other Lab renders failed on both sides, the control included: a 32-bit
  fail-fast (c0000409) that also fails `replay_composition.exe`
  (`test_composition_recording`, already failing before this change), access
  violations while another VM workload ran, and the shadows unit study's
  "required unit source admission incomplete". Separate follow-up.

**Busy save (1498 AD), census.** Every natural terrain chunk compacts (1,289
chunks, 3.36 M vertices at 56 B); mountain and decal chunks stay 92 B. At equal
vertex counts natural terrain vertices plus indices fall from 343 to about
224 MiB (−35%). Unique resident geometry: 823 → 750 MB at 4,608 owners (−8.9%);
about −8.6% later in the run at about 9,000 owners.

**Census trace.** With exact strides the census line outgrew the 1 KB trace
line and lost layers after 21, the natural layers among them; it now
continues on `geometry-census-more` lines (`test_geometry_census.py`).

**Test harness repairs.** Earlier changes today left stubs behind in
`test_durable_preparation`, `test_required_geometry_admission`,
`test_resident_mesh_ownership`, `test_zoom_mesh_cache` (residency
dependencies), `test_unit_reflection_selection` (route source cache, now one
publication per emit), `test_fresh_map_idle` (a completed render stays sampled
until its swap; the composition skips a repeated generation), and
`test_resource_instances` (explored fog keeps the live clock, section 54).
`residency_candidates.h` compares use times as 64-bit.

## 65. Where the busy save's GPU memory goes (October 10)

The memory census now counts the pulled pages (fresh_pipeline's batched
ground submission) and prepared water parameters, which it previously left
unattributed. Busy save during play (`memory-census`, MiB):

| Item | MiB | Note |
| --- | --- | --- |
| Tile geometry | 1,600-1,900 | the whole explored world, loaded at start |
| Pulled pages | 400-480 | GPU copies of visible ground vertex and index bytes (budget 512) |
| Retained composition | 310-370 | |
| Textures | 261 | |
| Static slots and overlays | 209 | |
| Shadow / scene targets / units | 100 / 92 / 94 | |

Adapter usage peaks near 3.7 GB. The pulled pages exist to draw ground and
water in a few calls instead of about 3,000 (about 3 us each); they copy each
occurrence's vertex bytes at the source stride, so they are a second copy of
the 168-byte ground layers. Two exact remedies, both picture-identical:
compact the ground vertex (pages copy at the compact stride), then, in R3,
pooled ground storage that both the input-assembler and pulled draws read
directly, removing the copy.

## 66. Stage 5, step 2: integrated ground meshes elide constant elements (October 10)

**Change.** An integrated (168-byte) mesh drops every input element whose
bytes are the same on all its vertices, position excepted
(`render_core/integrated_vertex.h`, `prepare_mesh`). The mesh keeps its full
168-byte constant row (`PreparedMesh::constants`), uploaded beside its vertices
and placed per chunk (`template_offset`); `elided` is the element mask. Input
layouts are derived per mask from the full ones: an elided element reads the
row through a stride-0 slot 1 (`integrated_layout`/`bind_integrated` for the
legacy and fresh draws, border layouts per mask). Pulled pages copy the row
after the vertices; the generated pulled shader reads an elided element from
it and skips it otherwise, with element sizes generated from the same table.
Ground, river, land and route meshes elide under the fidelity profile only,
where ground layers never cast shadows; waves stay 168. Consumers test the
mask before the stride, since a compact stride can equal a natural one. The
world backing codec stores the mask and row (format 6). The same
`C3X_RENDERER_COMPACT_VERTEX_CONTROL=1` keeps every mesh full.

**Verification.**
- `test_integrated_elision.py`: underlay- and river-like meshes packed both
  ways; every input of the main, border and caster layouts and of the pulled
  address helper (compiled as C++ from the generated shader text) fetches
  the full format's bytes; codec round trip; mutating the helper fails it.
- The GPU test compiles the generated pulled shaders on the VM's production
  shader pack (`C3X_RENDERER_GPU_TESTS=1 test_runtime_shader_programs`).
- Lab A/B (legacy draw path), byte-identical: grassland 3, floodplains 3,
  transitions 2, seas and oceans 2, rivers 2, infrastructure 3, shorelines
  detail and gameplay at four hours (23 pairs). Water scene geometry
  6.69 → 4.23 MB (−37%). The rivers fixture's lifecycle witness fails on
  both sides, independent of the format; its images match.
- The Lab cannot run the fresh pipeline (the sandbox client's unit prewarm
  and the reference build's trial configuration have drifted), so the fresh
  path was checked by its parts and in game: pulled draws 156,577 with 0
  fallbacks and 0 refusals, no draw or shader failures, frames normal at 1x
  and zoomed out.

**Busy save, in game.** Underlay vertices 60-80 bytes (about 73 on average,
−56%), routes 52-80, river 76-116. The geometry cache stays at its budget
(about 1.87 GB) while holding about 12,700 owners instead of about 9,000:
unique geometry per owner 161 KB (no compaction) → 137 KB (natural terrain,
section 64) → 104 KB. Pulled pages hold 6,604 entries in 380-410 MiB (was
3,057 entries in 470 MiB).

## 67. Memory by save, the fixed geometry tier and a page accounting fix (October 10)

**Whole-world residency, measured** (`world-gpu-residency` at load):

| Save | Map keys | World GPU bytes (required / admitted) | Adapter during play |
| --- | --- | --- | --- |
| Busy, 1498 AD | 16,900 (130x130) | 1,602 / 1,599 MB | 3.5-3.7 GB |
| User, 3350 BC | 398 | 56 / 56 MB | 1.8 GB |
| Light | 38 | 2 / 2 MB | 1.7 GB |

After both compaction steps the busy world is essentially whole-resident.
Small maps are dominated by fixed costs, not geometry. User save: static
raster slots and overlays 419 MB, retained composition 370 MB, unattributed
(driver, swap chain and other allocations) about 440 MB, textures 261 MB,
shadow 100 MB, scene targets 92 MB, geometry 72 MB.

**Fixed geometry tier.** In whole-world residency the geometry budget is now
at most the explored world's canonical content plus a fixed 384 MiB allowance
for view-specific content, and view-specific owners are evicted before
canonical ones (`geometry_tier`, `canonical_geometry_bytes`;
`C3X_RENDERER_GEOMETRY_TIER_CONTROL=1` keeps the free-memory budget alone).
Renderer64 draws zoomed-out views from canonical content, so view-specific
geometry is about zero today and on this VM the free-memory budget is the
smaller one: the busy zoom-out run is unchanged (owners, budget, adapter).
The tier guards larger-memory machines and future view content.
`test_required_geometry_admission` covers the eviction order.

**Accounting.** The cache charges owners' CPU metadata (dependency maps,
proofs, receipts) to the geometry budget: busy save 1,879 MB accounted for
1,474 MB of unique GPU buffers at load, 2,042 vs 1,599 MB in the residency
receipt, so loading refuses world keys early. Next: account GPU bytes and
CPU metadata separately.

**Pulled page accounting.** A run that drew one occurrence twice put it in
one page group twice; the second mapping released the new page before it was
counted, so resident bytes underflowed and trim retired every page each frame.
User save: 2,391 retirements and 519 MB of copies for 101 entries. Sources are
now de-duplicated by key before packing (`test_pulled_mesh_pages.py`, on the
VM's GPU; fails without the fix).

**Captures.** The helper's in-module teardown crash (offset 0x147b3) kills it
before a buffered trace is written; pass `C3X_RENDERER_TRACE_BUFFERED=0` in
`-RendererOptions` when the core log is needed.

## 68. Busy save pass ablation (October 10)

Busy save, `near`, cadence only, one run each (same build; the per-pass skips
are the sandbox's `C3X_SANDBOX_SKIP_*` switches):

| Run | 1x idle | 1x scroll x (fps / step p50 / adopt p50) | 2x scroll | 3x scroll | 3x idle | 1x idle, end |
| --- | --- | --- | --- | --- | --- | --- |
| Base | 44.2 fps | 5.9 / 260 / 191 ms | 4.0 / 454 / 340 | 4.0 / 354 / 256 | 55.9 | 26.7 |
| No water pass | 54.6 | 5.8 / 182 / 129 | 4.4 / 287 / 250 | 6.7 / 226 / 178 | 53.6 | 41.5 |
| No vegetation | 45.0 | 5.9 / 214 / 121 | 4.6 / 250 / 197 | 5.2 / 246 / 202 | 53.7 | 38.7 |

The reflection run failed to start (no session). Run-to-run noise is about
±30% on the scroll segments (the base run's 1x y segment: 31.1 fps, steps
157 ms), so only large effects count: the water pass costs about 10 fps of
busy 1x idle (it is the largest single GPU pass), vegetation little. Neither
pass explains the scroll steps: zoomed steps take 250-450 ms with or without
them, because a zoomed adoption draws the step twice (the canonical 1x image,
then the presented zoom).

## 69. A settled zoomed step is drawn once, at its presented zoom (October 11)

**Why zoomed steps were slow.** Zoomed in, every adopted step drew the
hidden canonical (1x) image before its presented-zoom draw. The canonical
lane's caches are cold while zoomed (its static raster refines at a quarter
budget and its retained water overlays do not match), so that draw redrew
every layer near water: the water phase took 36-80 ms against 3.5 ms at 1x
(busy save, trace level 2). A block-crossing job then redrew the canonical
image a second time at adoption.

**What reads the canonical image while zoomed.** With the zoomed draw ready,
the displayed frame does not sample it. The fallbacks that do (a new step
whose zoomed draw is still pending, the unprojected view, no projected
sampler) magnify it about the frame centre, so they read only the centred
rectangle the zoomed view covers. Civ III's commit-time quantize and copy read
the whole image, but their result is shown only by the native fallback
display (game loading, a failed retained graph). Zooming back to 1x draws a
fresh 1x frame.

**Change.** A deferred step adopted while the zoom is settled at its
destination (above 1x) is drawn once, at the presented zoom, and the
canonical centre is filled by minifying that draw
(`render_core/canonical_minify.h`: a box average of the zoomed pixels each
canonical pixel covers about the `SceneProjection` centre; exact at whole
zooms). That draw becomes the new map's first zoomed front instead of being
drawn again after publication. A full camera job (a block crossing) keeps its
canonical draw, but its adoption no longer redraws it at 1x while the zoom is
settled. `C3X_RENDERER_ZOOMED_STEP_CONTROL=1` restores the old path.

**Measured** (busy save, `near`, cadence, same build, control alternated;
one control run disturbed by the host and excluded):

| Step p50 | New (4 runs) | Old path (3 runs) |
| --- | --- | --- |
| 3x scroll | 153-189 ms, 21-28 steps in 5 s | 295-350 ms, 14-17 steps |
| 2x scroll | 253-491 ms (median about 300), 10-18 steps | 278-433 ms (median about 380), 7-12 steps |

1x is unchanged. With buffered traces, a covered zoomed step's draw now takes
26 ms at 2x and 18 ms at 3x (it was the full canonical render). 2x gains
less because about a quarter of its steps are block crossings, which still
run full camera jobs (shadow rebuild 40-110 ms, membership 32 ms p50).
Window frames during zoomed scrolling and after zooming back out are complete
and sharp.

**Tests.** `test_zoomed_step_canonical.py`: on the VM's GPU, a zoomed image
built by replicating canonical pixels over their projection footprints
minifies back to exactly those pixels at 2x and 3x, and nothing outside the
covered rectangle is written (a one-pixel centre error fails it); a source
contract for the settled-zoom draw, the first-front hand-off and the skipped
1x redraw (fails on the previous code).

## 70. A block crossing re-registered every river page it entered (October 11)

**Where crossing proofs went.** At a block crossing the shadow atlas
registers its entering casters' dependencies (busy save, trace level 2, per
crossing p50): about 250 new producers, 1,543 river-source expansions and
854,000 dependency watches, of which 836,000 came from expanding river
sources; 38 ms of registration (p90 96 ms). River page inputs are shared per
page in memory, but decoded and recompiled pages are distinct objects with
equal content, and sources were keyed by object.

**Change** (exact; each producer keeps the same normalized keys):
- River page inputs equal in content (values and flow) are one source for
  registration (`RendererState::shared_river_inputs`, a content hash memoised
  on `PageInputs`, weakly held);
  `C3X_RENDERER_PROOF_SHARING_CONTROL=1` turns this off.
- The atlas adds each producer once per mesh generation instead of once per
  caster (about 7 casters per generation).
- Keys are emitted in domain order and normalization skips its sort when a
  list is already ordered.

**Measured** (busy `near`, trace level 2, one run each; per crossing):

| | Old path | Sharing | Sharing + once per generation |
| --- | --- | --- | --- |
| Source expansions | 1,543 | 4 | 4 |
| Dependency watches | 854k | 16k | 26k |
| Shadow build, p50 / p90 | 63 / 120 ms | 53 / 94 ms | 48 / 92 ms |

Registration itself fell to 12-19 ms; the rest of the crossing's shadow
build is caster refresh (10-12 ms), page proofs and selection, and the page
draws. `static-validation-counts` now reports `atlas_source_watches`,
`atlas_normalize_ms` and `atlas_source_ms`.

**Tests.** `test_static_dependency_reuse.py`: distinct owners with equal
river inputs expand once in the production `ShadowCasterProofs` and an edit
to a shared node still reaches every producer that uses it (fails without the
sharing); registration iterates unique generations (source contract); the
extracted-registration fixtures include the sharing helper.

## 71. Scrolling showed one frame per Civ III tick (October 11)

**Cause.** A scroll step starts a camera job, and a job that moves the camera
retires the retained scene view, so display frames hold until the step is
adopted. Since 2c.1 most steps are deferred: the job completes at once and
draws nothing until Civ III's next tick adopts it, so for about 50 ms of every
78 ms tick the worker sat idle with the view frozen. Light save, `near`: 60 fps
idle, 15.6-18.2 fps while scrolling, steps at one tick (78 ms).

**Change.** A deferred step leaves the live scene as the shown one, so after
its completion display frames keep animating that view and unit travel
continues; the step's adoption then takes its frame's units at the current
visual time (a frame shown meanwhile selected units for the old view, so an
entering unit would otherwise miss the step's first frame). A frame must end before
Civ III's next tick adopts the step (about 60 ms after the request, measured):
a frame starts only if the median worker time of recent ambient frames (no
newly committed front, settled zoom) ends it within 55 ms of the request.
`C3X_RENDERER_DEFERRED_HOLD_CONTROL=1` always holds.

**Measured first with the hold released unconditionally** (same build,
control alternated):

| | Released always | Held |
| --- | --- | --- |
| Light, scroll fps (1x x / 2x / 3x) | 31.6 / 27.0 / 27.7 | 18.2 / 15.7 / 15.6 |
| Light, steps | 78 ms | 78 ms |
| Busy (released unconditionally), adoptions in the scroll segments (traced) | 71 | 85 |
| Busy, camera-adopt queue p90 | 157 ms | 85 ms |

On the busy save a display frame costs the worker 13 ms (p50, p90 30 ms;
light 2.3 / 4.4 ms), and frames started during the hold delayed adoptions:
16% fewer steps when the hold was released unconditionally. Gating on the
frame cost instead (below 8 or 10 ms) was tried and dropped: one zoom frame's
cost closed the gate, and with frames held it was never re-measured (light
scrolling 16-24 fps). With the adoption deadline (same build, control
alternated):

| | Deadline | Held |
| --- | --- | --- |
| Light, scroll fps (1x x / 2x / 3x), two runs | 27.8-29.5 / 24.3-25.0 / 23.7-27.2 | 15-18 |
| Busy, steps in the four scroll segments, two runs | 79, 65 | 65, 58 |
| Busy, 1x idle | 45-50 fps | 44-48 fps |

## 72. Rivers drew their whole tiles every frame (October 11)

**Where busy idle GPU time goes** (`C3X_RENDERER_PROFILE=4`, drained marks
inside the water span; busy save, 1x idle, p50 per frame of 54 ms):
the river layer 21.7 ms, units 5.4, interface evaluation 4.7, reflections
4.5, assembly 3.0, the sea surface 2.2, the retained near-water overlay
composite 1.3. At 3x idle the river layer is 5.4 ms of 34.

**Cause.** The river layer of every tile with or near a river is the tile's
whole ground grid (32 x 32 cells, 2,048 triangles at gameplay tile size), and
the river shader decides that a pixel lies beyond its bank only after about
fifteen texture samples, then discards it. Rivers run along tile edges, so
most of each river tile, and nearly all of a tile that is only near a river,
is shaded and thrown away every frame.

**Change** (exact). Every river shader variant discards a pixel at least
20.7 screen pixels (1x) from the channel, and the continuous-river material
at 11.7 plus its bank antialiasing width max(fwidth(distance), .35) (grit at
14.7). The ground compiler leaves out a river triangle whose three vertices
are all beyond that reach (one pixel of margin), with the width taken from the
triangle's own screen-space distance gradient (world ground projects as
`project_world_content` does) at the smallest presented zoom, and keeps only
the vertices the remaining triangles use. Fidelity profile only;
`C3X_RENDERER_RIVER_CULL_CONTROL=1` keeps whole grids; the flag is part of
the ground recipe key.

**Verified.**
- Lab, byte-identical with and without the cull: every rivers and shorelines
  case at every recipe hour. With the reach deliberately cut to 4 px the
  rivers image differs in 3,321 pixels, so the comparison exercises the cull.
  (The rivers fixture's water-lifecycle witness fails on both sides, as before.)
- `test_river_reach_cull.py`: the bound against the shader's own discard
  arithmetic for 20,000 random noise samples and widths, with the shader
  constants it relies on asserted; the compiler and recipe key contract.
- In game (busy, drained GPU marks, p50 per frame):

| Busy | River layer | GPU frame |
| --- | --- | --- |
| 1x idle | 21.7 → 13.8 ms | 54.0 → 50.5 ms |
| 1x idle, last segment | 20.8 → 12.5 ms | 59.2 → 50.6 ms |
| 3x idle | 5.4 → 5.1 ms | 34.4 → 30.7 ms |

The floor keeps the widest (legacy, 20.7 px) variant; every shipped pack uses
the continuous or static-optics material, so a 15.7 px floor would also be
exact for them, but it would tie the geometry to the pack's shader variant.
Cadence runs (idle 39-49 fps either way) are within this VM's noise; the
remaining river cost is the corridor the shader really draws.

Diagnostics added with it: drained GPU marks inside the water span
(`water_surface_water`, `_river`, `_overlay`, `water_city_effects` and the
dynamic spans, `C3X_RENDERER_PROFILE=4`), water lighting cache builds and
reuses in `fresh-overlay-layer`, and the step render's unit asset wait
(`units_ms`, `unit_waits` in `render-setup-phases`: about zero after load).

## 73. A zoomed block crossing is drawn at its zoom too (October 11)

A camera job that moves the camera retires its retained view, so no display
frame runs until its adoption. Settled zoomed in, such a job (a block
crossing) is now drawn at the presented zoom with the canonical centre
minified from it, as deferred steps are (section 69); a same-view content job
keeps its canonical draw, since display frames run during it. Busy save,
trace level 2, crossing jobs: at 2x 153 ms p50 with a 1.9 ms water phase
(the cold canonical lane's water phase was 36-80 ms), at 3x 98 ms. Cadence
runs at this hour were too disturbed by the host to compare step periods
(one run showed 28 fps at 3x idle). Window frames during zoomed scrolling and
the zoom-out transition are complete. `C3X_RENDERER_ZOOMED_STEP_CONTROL=1`
restores the canonical draw for both.

## 74. The fresh pipeline draws from the resident world (October 11)

Whole-world restructure, increments 3 and 4 (camera decoupling design,
stage 6), behind `C3X_RENDERER_RESIDENT_WORLD=1`.

**Draw reads.** With the store on, the pipeline's contributor queries read
the resident world (`ResidentWorld::query`, wrapped turns, the mirrored pass's
height extents) instead of the camera window's lease and region index; a
capture no longer rebuilds the region index or the resident tile set, and
`view_revision()` is the store's revision, which moves only when stored
content changes. Records are in the world basis (resident basis 0). Window
records that still come from the camera assembly (resource poses, aquatic
bounds, waves) use the window basis.

**Shadow casters.** The caster selection is built from the store's tiles
under the camera window box (each canonical tile once, in occurrence order,
owning its meshes) instead of the window's lease, so caster footprints are in
the world basis and the static repair converts them with basis 0. The caster
set is unchanged.

**Loading the store.** On the first gameplay frame of a scope, every
explored tile whose resident content passes the window's own admission test
enters the store from the retained world (`resident_world_load`): busy save,
8,450 tiles, 37,004 records, 77-96 ms once, none refused. Over whole runs no
camera assembly then rewrote a stored tile (0 rewrites), so a crossing no
longer changes the store; only evictions do (the VM's geometry budget). A
preload during the loading-time preparation was tried first: its frames have
another view height, which the projections depend on, so the store's scope
discarded it.

**Moved-window steps are deferred** (increment 5a). A step whose window box
moves used to be a full camera job, then drawn again at adoption. With the
store, its Civ III ownership can come from the store instead of the last drawn
window: every window tile must be stored from its current content (a content
key per stored tile), and its flags are its content's replacement flags (one
helper now shared with the render loop) plus its owner's resource
replacement. The step then completes at once and is drawn once, at adoption.
Witness run: 20 moved steps deferred; the adoption's promised-against-drawn
ownership check (`deferred-step-ownership`) found no disagreement. The
remaining refusals were evicted tiles (9-13 missing) and steps that came after
an idle prefetch render, which recomputed the per-render store flag as off:
the draw and the deferral now read a flag only gameplay renders set.

**No window draw records** (5b). While the store drives the draw, a camera
assembly keeps the window's anchors, footprints, use stamps and topology
attachments but builds no draw records, and a stored tile built from the same
owners and content is not rebuilt (13,202 skips in a busy run). With the
window's revision no longer following the window, the window basis (resource
poses, aquatic bounds, waves) is taken from each capture's view; the witness,
which keeps the records, found it equal to the tracked basis in all 208
captures. A crossing's assembly fell from 21.6 to 14.3 ms (p50, trace level
2); a covered step's render setup is about 9 ms.

**Crossing shadows.** The largest remaining crossing cost is the shadow
rebuild when the caster set changes (busy, trace level 2, p50 of crossing
builds): caster refresh 9-11 ms, atlas proof registration 9.5 ms, caster
selection 3.8 ms, page draws 6-7 ms (16-24 pages); with the moved-step
deferral it runs inside Civ III's synchronous adoption (1x adopt p90 186 ms).
Two changes were tried:
- The caster scope keyed the store's global revision, which evictions
  anywhere bump; it now keys the revisions of the store blocks under the
  caster box (`ResidentWorld::region_revision`). Correct, but not the cause:
  caster-set rebuilds stayed at 22-25 per run.
- A caster box one block wider than the window, kept while the window stays
  inside it (`caster-scope` trace): the caster set changed only 21 times in a
  run (the box moving two blocks at a time), but each refresh cost about 50%
  more (17-21 ms) and unfinished builds repeat, so shadow work did not fall
  (2x adopt p50 42 against 28 ms). Reverted to the window box. The fix is
  caster collection cached per block, so a change costs its blocks only.

**Crossings without the window's costs** (store on):
- *Casters per store block.* A block's casters and keys are collected once
  per block generation (`world_casters`); a refresh reuses 84% of blocks.
  Caster refresh at 1x crossings fell to 4.7-16 ms; most of what remains is
  copying and keying the whole set (about 12,600 casters with wrap copies:
  selection 6.9, dedupe 2.5, journal 4.0 ms p50).
- *No retirement at a window change.* A crossing's render retired the
  pipeline's whole selection (capture, region of interest, lights, shadow
  casters and placement groups) because the pipeline borrowed the window's
  records; with the store it owns its meshes, so a window change keeps them.
  The caster witness found the reason: every crossing started from no
  caster selection.
- *The caster refresh leaves the crossing.* A scene render (a camera job or
  an adoption, which Civ III waits for) whose window moved by at most a
  block keeps the drawn casters once, since the new view's casters lie
  inside the previous box (8 tiles past the view); the next frame collects
  the new set. A far jump refreshes at once; `C3X_RENDERER_CASTER_DEFER=0`
  always does. In a run, 27 refreshes were deferred, 22 of them then ran in
  display frames.

Crossing renders (busy, trace level 2, p50): 52 ms with the store against
101 ms for the window (shadows 0 against 40 ms, preparation 14 against
43 ms). Witnesses unchanged: 225 selections, 225 bases, 34 caster sets.

**Measured with the crossing changes** (cadence, three runs per side
alternated, same build; the VM's run-to-run variance is large, so totals
matter more than single segments):

| Busy step p50 (steps) | Store | Window |
| --- | --- | --- |
| 1x scroll x | 157-176 ms (29-31) | 161-169 ms (31-33) |
| 1x scroll y | 155 ms each run | 160-228 ms |
| 2x scroll | 140-155 ms (30-34) | 150-158 ms (28-34) |
| 3x scroll | 79-80 ms (45-50) | 80-83 ms (42-47) |
| Scroll steps in all | 121, 124, 129 | 116, 128, 121 |

Light save: 78 ms steps either way; display frames while scrolling
40 / 38 / 37 fps with the store against 34 / 31 / 33 (1x / 2x / 3x). The store
is slightly ahead overall and steadier, and slightly behind at busy 1x
horizontal scrolling.

**Tried and reverted: a view revision local to the view.** Keying the
view's caches by the store blocks under the camera window (as the caster
scope does) instead of the whole store's revision rebuilt the capture 42
times in a busy run against 26 (`fresh-scene-phases resident_builds`): once
the store is loaded a crossing inserts nothing, so the global revision stays
put across crossings and moves only when the world changes, while a local key
moves with the window.

**Validation across scenarios** (store on, witness on, one run each):

| Scenario, save | Selections equal | Notes |
| --- | --- | --- |
| `units`, busy | 612 of 612 | 2 moved steps deferred, no failures |
| `zoom-out`, busy | 105 of 155 | the 50 others have extra store contributors (up to 1,900), never missing ones: zoomed out the view outgrows the window, which the store covers; frames complete to the edges |
| `reveal-scroll`, 4000 BC | 61 of 61 | 57 stored tiles rewritten as reveals and a new city changed them; two retained-memory refusals, also in window mode |
| `turn`, start save | 1 of 1 | the scenario's research check fails in window mode too |

Caster sets contained the window's in every check, the window basis matched
throughout, and no deferred step's ownership disagreed with its draw.

**Moved-step deferral, measured again** (store on, cadence, three runs
each, scroll steps):

| | 1x horizontal | 2x | 3x | All segments |
| --- | --- | --- | --- | --- |
| Deferred | 25, 22, 30 | 31, 35, 34 | 47, 45, 50 | 118, 117, 129 |
| Full jobs | 29, 27, 29 | 21, 18, 33 | 39, 39, 45 | 104, 99, 122 |

Deferral stays on: it wins zoomed in, and the 1x difference is within the
run-to-run spread.

**Default on.** The resident world now drives the draw by default;
`C3X_RENDERER_RESIDENT_WORLD=0` keeps the window path. With the default (no
option set), witness runs of the busy `near`, `combat` and `city` scenarios
were clean (selections, casters, feed, ownership); the city scenario's
retained-memory refusal also happens on the window path. The Lab renders
through the 32-bit candidate DLL, which has no fresh pipeline, so its images
cannot change.

**What a crossing's window assembly still costs** (store default, busy,
trace level 2, `resident-world` trace `build_ms assembly_ms tail_ms`): the
per-tile build loop 6.6 ms p50 (60 ms p90, when tiles compile), the assembly
2.1 ms, and a 5.3 ms tail, mostly copying about 4,000 tile records into the
window cache. Display frames during busy 1x horizontal scrolling (about
7 fps on either path) are presented from Civ III's loop, about once a tick,
while it waits in each step's synchronous adoption, so they follow the
per-step cost rather than a separate hold.

**The window basis across the wrap seam.** With no window draw records the
window basis was taken from each capture's view; across the wrap seam Civ III
re-bases the camera a world width while the window keeps its frame, so in 4
of 235 captures that basis was a world width (8,320 px) off, which would
place resource poses and waves off screen. The basis now follows a renderer
serial bumped whenever the window changes (`window_frame_serial`); the
witness then found it equal to the tracked basis in all 222 captures.

**Store layout.** A tile keeps one record array with layer offsets instead
of two vectors per layer (geometry has about 40 layers): the busy save's
store fell from 20.8 MB to 5.0 MB for about 28,000 records.

**Witnesses** (`C3X_RENDERER_RESIDENT_WORLD_CHECK=1`, busy save, `near`):

| Check | Result |
| --- | --- |
| Every window record has its store copy (feed) | 0 missing, 0 differing, one offset, all runs |
| Visible selection, store against the window's by exact test | equal in 120, 133 and 134 captures (no extra, missing or reordered contributor) |
| Shadow casters, store against window, by key | equal in all 29-46 refreshes per run |
| Window basis against the tracked one | equal in all 208 and 222 captures (after the seam fix below) |
| Promised ownership of deferred steps against the adoption's draw | no disagreement (20-27 moved steps per run) |

**Measured** (cadence, same build, control alternated; increment 3):

| Busy step p50 (steps) | Store | Window |
| --- | --- | --- |
| 1x scroll x | 171 ms (27) | 170 ms (29), 157 ms (30) |
| 2x scroll | 158 ms (27) | 169 ms (20), 151 ms (33) |
| 3x scroll | 155 ms (33) | 152 ms (34), 81 ms (44) |

Light save: 78 ms steps either way (73 and 74 steps at 1x). The store is
neutral so far, as expected: crossings still build the camera window.

**Measured with all of the above** (store against window, cadence, busy
alternated, same build):

| Busy step p50 (steps) | Store | Window |
| --- | --- | --- |
| 1x scroll x | 215 ms (25), 206 ms (25) | 160 ms (31), 165 ms (32) |
| 1x scroll y | 155 ms (15), 155 ms (15) | 159 ms (15), 233 ms (15) |
| 2x scroll | 105 ms (36), 81 ms (38) | 153 ms (29), 155 ms (29) |
| 3x scroll | 81 ms (44), 80 ms (48) | 236 ms (21), 149 ms (35) |

Zoomed in, busy steps now take one Civ III tick (78 ms) with the store, a
first on this save. 1x horizontal scrolling, which crosses a block every few
steps, is slower: a crossing still assembles the window and replaces the
caster set, now inside the adoption. Light save: 78 ms steps either way;
display frames while scrolling 47 / 40 / 37 fps with the store against
38 / 34 / 34 (1x / 2x / 3x). Deferring moved steps alone (store on, the
deferral switched by `C3X_RENDERER_STORE_MOVED_STEPS`) was neutral within the
noise: 104-123 scroll steps per run with it, 104-115 without.

**Where a busy step goes** (trace level 2, store on). A covered step's
adoption draw now takes about 25 ms (render setup about 10 ms, p50), and a
crossing's 50-125 ms (membership assembly 26 ms p50, shadows 15-47 ms). The
request itself waits for Civ III's next tick: the helper completes the
deferred step within 4 ms and draws it at adoption, about 60 ms after the
request, so a busy step still spans two ticks when the adoption draw, Civ
III's image batches and the transport waits exceed one (review 61-62).
Crossing adoptions are the long ones (camera-adopt service 44-240 ms in a
traced run), which increment 5 removes.

**Tests.** `test_resident_world_pipeline.py`: the extracted production
`contributors()` returns the same records in the same order from the store as
from the window's lease and region index on equal data, across the wrap seam
and in the mirrored pass (fails without the wrap shift or the mirrored
extent); the caster selection takes each canonical tile under the window box
once in occurrence order and owns its meshes (fails without the wrap or the
per-tile dedupe). `test_resident_world.py` covers the flat tile layout
(layers added in any order, refused inconsistent layouts);
`test_zoom_mesh_cache.py` checks that a store-only append changes no window
records, anchors, footprints, use stamps or prefetch accounting, and keys the
tile by its content. A moved window's ownership from stored content maps to
Civ III's order, adds the owner's resource replacement, ignores placement
and visibility flags, and refuses changed content, a missing tile or unready
site art (fails without the content key or the resource bit); a source
contract keeps prefetch renders from switching the draw off the store (fails
on the previous pipeline). Casters are collected once per store block
generation: an edit or an entering block recollects only that block, a
leaving block's meshes are released (fails with the cache bypassed); a scene
render keeps the drawn casters for a one-block move once, display frames,
repeats and far jumps refresh (fails without the adjacency test); a source
contract that store mode keeps the selection at window changes (fails on the
previous code). The window basis keeps its frame when the camera
re-bases across the seam (fails with the per-capture basis); region
revisions change only for blocks under the region, across the seam; the
caster box with and without a margin.

## 75. Smoke and flames stay visible through a zoom (October 11)

**Cause** (reported by the user). A zoom, and the first frames after a jump,
show static previews: the new zoom's static layer is still being drawn, so the
frame resamples the cached image (color and depth) from another zoom. The
attached effect layer (flames, smoke, night lights, volcano plumes) was
skipped on every preview, because tested against the resampled depth these
effects had flashed as bright bursts. A zoom is many previews, so the effects
vanished for the whole zoom and returned once the camera settled.

**Change.** Previews draw the effect layer without the resampled depth. Its
quads face the camera above their own bodies and its shaders read no scene
depth. `C3X_RENDERER_PREVIEW_EFFECTS=0` skips them on previews again.

**Checked** (busy save, `near`, 10 frames/s): mid-zoom preview frames keep
every city's smoke columns, with no bursts; the first frame after a minimap
jump shows smoke at once. Test: `test_preview_effects.py` (source contract,
fails on the previous pipeline; the effect shaders must keep reading no scene
depth).

## 76. Stage 5, step 2: 16-bit natural and ground vertex elements (October 11)

**Change** (approved by the user on October 11; not byte-identical).
- **Natural terrain, decals and mountains** (world-projected meshes):
  - The normal is stored as SNORM16x4.
  - The four material channels, relief height/blend and the volcano owner are
    half floats.
  - The input assembler expands them, so no shader changed.
  - World xyz+valid, position, uv and base terrain stay 32-bit. Base terrain
    is a class code plus coverage (42 + coverage needs a float).
  - Strides: compact 56 → 40 B, full 92 → 68 B (`natural_offsets` in
    `render_core/prepared_mesh.h` describes all four formats).
  - Every consumer derives its layout from that table: the terrain, mountain
    and object layouts, both shadow caster layouts (coverage is the desert
    channel, read as R16G16_FLOAT) and territory borders.
- **Integrated ground, rivers and routes** (meshes that elide constants):
  - Bit 0 of the format mask (`integrated_quantized`; position is never
    elided) marks half-float vector elements: normal, visibility, material
    weights, relief, river, hydrology and owner/flow.
  - Position, uv, macro uv, world and the one-float elements stay 32-bit.
    Elided elements still read the 32-bit constant row.
  - Input layouts switch those elements to R16G16/R16G16B16A16_FLOAT. The
    generated pulled shader decodes them with `f16tof32` and computes packed
    addresses from the half sizes.
- **Fallback.** A mesh with any value outside the 16-bit range (a normal
  component beyond ±1, or a magnitude over 65,504) keeps its exact 32-bit
  format.
- **Other changes.** The world backing codec is version 7, and the terrain
  and ground keys include the switch. `C3X_RENDERER_QUANTIZE_CONTROL=1` keeps
  the exact formats.

**Busy save, in game** (whole world resident, census against the control,
same build):

| | Control | Quantized |
| --- | --- | --- |
| Required world GPU bytes | 1,554 MB | 1,267 MB (−18%) |
| Natural terrain / decals / mountains (logical) | 275 / 257 / 514 MB | 210 / 198 / 399 MB |
| Underlay / river / route | 136 / 30 / 29 MB | 116 / 23 / 26 MB |
| Geometry cache (budget-bound) | 1,822 MB, 14,320 owners | 1,617 MB, 15,916 owners |
| Pulled pages | 411 MB | 356 MB |
| Adapter | 3,281 MB | 3,061 MB |

Every natural chunk quantized: terrain is all 40 B, and decals and mountains
are all 68 B. Underlay strides are 52–72 B, down from 60–84 B. Neither run
had native failures or shader errors, and pulled draws ran without fallbacks.
Zoom and scroll cadence were within run noise.

**Verification.**
- `test_natural_compact_vertices.py`:
  - Packs the same vertices in all four natural formats.
  - Every main, caster and border input must equal the full format's bits,
    or for 16-bit inputs the input assembler's conversion of the exact value.
    The nearest half is computed from first principles, ties to even.
  - Class codes (1, 1.25, 2, 2.5, 5-8, 42) stay exact.
  - Out-of-range values keep the exact format.
- `test_integrated_elision.py`:
  - Quantized and exact meshes go through the main, border and caster
    layouts and the pulled address helper.
  - A quantized mesh with no elided element keeps its row (mask 1).
  - Codec round trip.
- Mutations fail both tests: a truncating half encoder, and a pulled helper
  that reads position from the row when bit 0 is set.
- `test_runtime_shader_programs` (GPU, VM) compiles the generated pulled
  shaders against the production pack.

**Lab A/B** (quantized against the control, legacy draw path, 32 pairs):
- Coverage: grassland 3, plains 2, desert 2, floodplains 3, transitions 2,
  hills 2, mountains 3, volcanoes 4 (detail and active at 128 and 224),
  shorelines 4 (detail at four hours), seas and oceans 2, rivers 2,
  infrastructure 3.
- Result: 2 pairs are identical. Of the 30 that differ, the largest channel
  difference is 1 level in 24, 2 in 3, 3 in 1 and 4 in 2. Pixels differing
  by more than 2 levels: none, except 0.001% in the two volcano renders at
  224. PSNR 73–85 dB.
- No reference image was replaced.
- The rivers gameplay fixture exits with code 1 on both sides; its images
  were still compared.
- Not run:
  - Borders: the Lab border preview needs PIL, which the Mac interpreter
    lacks. Border layouts read only position and world, which keep their
    offsets.
  - Forests: vegetation bodies are not quantized.

## 77. What a per-frame zoom draw costs (October 11)

Plan item 4 first measures a full draw at the current zoom on the busy save.
`C3X_RENDERER_STATIC_LEGACY=1` draws the visible static raster synchronously
whenever the zoom changes, as a per-frame zoom draw would.

**Cadence** (`near`, two runs per side, `zoom_report.py`). Zoom transitions
with cached previews ran at 16–63 fps, typically 25–35. With a full draw
every frame they ran at 4.5–19 fps; 1x idle was 36 fps against 55.

**GPU per zoom-changing frame** (`C3X_RENDERER_PROFILE=4`, isolated phases,
p50; paired with each frame's `static-compose` zoom):

| | Full draw (legacy) | Previews (default) |
| --- | --- | --- |
| Frame | 228 ms | 71 ms |
| Static | 146 ms | 2 ms (p90 158: refinement after the zoom) |
| Reflection | 25 ms | 2.6 ms |
| Water surface | 12.7 ms | 5.1 ms |

**By layer** (`C3X_SANDBOX_PROFILE_COMPLETION=1`, which now also writes a
`sandbox-completion` trace; only draws over 1 ms are reported, serialized,
mean per zoom-changing frame):
- **Water mirror redrawn at the new zoom: 85 ms.** Vegetation bodies
  59 ms, cliffs 12, features 10, cities 4.
- **Main static draw: 31 ms.** Vegetation bodies 18 ms, cities 4, cliffs
  3.6, ground (pulled) 3, features 2. Natural terrain is under 1 ms per
  draw.

**Draw calls are not the limit.** The static pass issues about 100 draws for
a full draw (ground is pulled; rigid parts are ordered packets). The cost is
shading, mostly vegetation, and it is paid twice when the mirror is redrawn.

**Consequence for item 4.** A per-frame draw that keeps the mirror until the
zoom settles costs about the main static draw: about 31 ms of isolated GPU
time in this VM, against 146 ms for the full redraw.

**Per-frame zoom drawing, measured** (`C3X_RENDERER_ZOOM_DIRECT=1`, busy save,
`near`, two runs per side). Moving zoom frames, and the frame that reaches
the destination, draw the visible static raster at their own zoom, and the
water mirror waits for the zoom to settle (`test_zoom_direct_draw.py`).
- Direct: transitions at 2.5–19 fps (mostly 5–11), and the first moved
  frame came 30–820 ms after the wheel.
- Previews: 8–50 fps (mostly 20–32), with the first moved frame 30–155 ms
  after the wheel.
- Direct runs presented 37–39 direct frames and 2–3 previews; preview runs
  presented 88–113 previews.

In this VM the direct draw is far slower than the stretched preview. The user
(October 11) cares about what disappears or flickers during the stretch, not
the stretch itself, so previews stay the default and the switch stays off.

**Preview consistency, first look** (default build, 10 frames/s window
samples during the `near` zooms). Within one zoom, previews alternate
between two sources for near-water overlays (roads, mines, fields and
features on coastal and river tiles):
- the source raster's retained overlay layer, resampled with it;
- a live draw at the current zoom against the resampled depth, when any
  preview source has no layer: a bootstrap, or a slot just promoted by
  refinement.

At a sampled switch at one zoom and camera (frames 742 and 743, 1.25x),
only animated units and smoke changed. A transient detector (a spot that
differs in one frame from both neighbours at the same zoom and camera) found
only scattered 200–1,400-pixel changes, which are animation. No sampled frame
lost roads, mines or smoke. The window samples catch about one frame in
three, so single-frame dropouts may still be missed.

A second capture covered the whole scenario, including the zoom out from 3x
to 1x. Its previews mix the old raster with the home raster or a bootstrap,
and draw the overlays live (frames 1007-1021). Consecutive same-zoom frames
changed only through animation: units, smoke, and water glints of up to
27,000 pixels on the sea. Neither the switch to the bootstrap at 1x nor the
first exact frame lost roads, mines or smoke.

## 78. Roads on rivers vanished on zoom previews (October 11)

**Report** (the user's recording, busy save at 2-3x near Nagoya). During zoom
transitions the two roads crossing a river vanished, then returned once the
zoom settled; mines, resources and shadows also seemed to change.

**Cause.** A frame is the static image plus a live layer composited over it.
An exact frame draws the animated water into the live layer, then composites
the retained near-water overlay layer (roads, bridges, improvements on river
and coast tiles) over that water. A preview baked the sources' overlay
layers into the static image instead, under the live layer. Its water is
also biased four pixel-depth units toward the camera, against the resampled
depth. Wherever water drew, it covered the overlays, so roads lying on a
river disappeared on every such preview. These were 25-30% of preview frames
on the busy save.

**Change.** Previews keep the overlay sources and composite them in the water
pass, after the live water and river and before the waves (the exact frame's
order). They are shifted by the same four pixel-depth units as the water.
Test: `test_preview_overlay_order.py` (fails on the previous pipeline).

**Checked** (busy save, `near`, 10 frames/s, the same crossing below the
mountain mine). Every overlay-layer preview, at 2.5x, 2x, 1.75x, 1.5x and
1.25x, keeps both road strips across the river, the two bridges and the mine.
No compositing failures.

**Still open from the recording.** These are not this bug:
- Coal lumps beside a mine gain their ground shadows only in the settled
  frame. Previews resample rasters drawn under another zoom's shadow field.
- Mines change size slightly between preview frames. Each preview is
  stretched from a raster drawn at a nearby zoom.

## 79. Coal piles flickered through zoom previews (October 11)

**Report.** In the user's recording, resources beside mines (a coal pile on a
hill) changed look during zooms; the user suspected shadow pops.

**Measured** (busy save, `near`, 10 frames/s, near-black share of a
fixed-world crop around the coal mine north-west of Nagoya). Settled frames
read 3.6% (1x) to 6.1% (2x). On every run, 12-14 preview frames read 8-10%:
the whole pile, including the parts buried in the hill, for one to three
frames per zoom step.

**Not the cause** (each measured):
- Shadows. Ground shadows are equal across zooms.
- Resolution. The 1x Lab and game renders match the 2x render downsampled.
- Per-frame zoom drawing order.
- The caster deferral (`C3X_RENDERER_CASTER_DEFER=0`).
- The resident-world restructure (`C3X_RENDERER_RESIDENT_WORLD=0` showed the
  same frames, so the behavior predates it).
- The overlay order of section 78 (`C3X_RENDERER_OVERLAY_CACHE=0`).

**Cause.** The pile counts as near-water, so it lives in the slot's retained
overlay layer. Every dark frame (14 of 14) was a preview that drew the
near-water overlays live: either a bootstrap, or a lane slot just marked
stale. The live path draws them with the preview water's four pixel-depth
bias toward the camera, which pushes buried rock through the hill. A stale
slot lost its layer because `invalidate()` bumps the slot revision without
touching its pixels, and previews required equal revisions.

**Change.**
- Slots count their invalidations. A preview keeps a slot's layer while every
  revision since the layer was written was an invalidation
  (`overlay_matches_pixels`).
- Previews whose sources all map one-to-one at whole-pixel offsets (a
  bootstrap at this zoom) have exact depth. They draw the water and overlays
  without the bias.
- Test: `test_preview_overlay_staleness.py` (the revision bookkeeping run in
  C++; fails on the previous pipeline).

**Checked.** Dark preview frames: 0 (previously 12-14 per run). Previews read
0.9-5.0%, within the settled range, and 1x bootstrap previews match the
settled 1x frame exactly. Zoom frame rates are unchanged.

**Early refinement** (option 1, `C3X_RENDERER_ZOOM_EARLY=1`, off by
default). The zoom lane refines toward the destination while the zoom moves,
with the shadow field fitted to the destination at once. Before the cause
was found it removed only some dark frames (7 against 12), and it roughly
halved zoom transition rates in the VM: 9.4-13 fps against 14-50, and
settling took 310-440 ms instead of 120-290 ms.

## 80. Native windows stopped updating: the composition budget (October 17)

**Report** (the user's `CAPTURE_GAMEPLAY.bat` session, a new game at
2240x1192).
- City view: the close button did not highlight on hover, though it still
  closed when clicked.
- Diplomacy with Germany: the first attempt to negotiate ignored clicks; the
  second reached the negotiation screen, where no choice took effect.

**Cause.** The input reached Civ III; the screen stopped showing its UI.
Each time a full-screen native window opened (bursts at 15, 35, 65, 80, 110
and 135 s), the retained composition refused the window's capture:
`retained-admission-rejected site=capture-output requested=10680320`
(`owned-pair` 21360640 once), at resident 255-267 MB against the fixed
256 MiB cap. It then logged `visual-failure-memory` and 200-420
`failed-retained-node` lines. Map hover and click handling stopped during
those windows, so the map did not take the clicks.

**What holds the budget** (new breakdown in the refusal trace; scripted
`city` scenario on the new-game save reproduced it with exit 1):
- 19 window-sized textures, 203 MB: 6 native uploads, 7 pinned by the
  composition being built, 2 fused sources, 2 retired native pixels, and 2
  outside the reachable graph (the HUD pair).
- Replay scratch 45 MB, smaller textures 11 MB.

All of these are in use. At 2240x1192 a window-sized texture is 10.7 MB, so
256 MiB fits about 24, and a new full-screen window needs a few more. At 4K
(33 MB each) no fixed cap of this size works.

**Change.** The budget is 40 window-sized textures, never below 256 MiB,
and never shrinks when the front is reset. That is 427 MB at 2240x1192 and
332 MB at 1920x1080; smaller windows keep 256 MiB
(`retained_composition.h`, `resident_budget()`; `budget()` exposes it).
- Test: `test_retained_budget_scaling.py`, which reproduces the observed
  267 MB + 21.4 MB request and fails on the previous code.
- `test_retained_view_lifetime` now bounds by the actual budget.

**Checked.** The same `city` scenario passes: zero refusals, zero failed
nodes, and the city view displays and updates at both city zooms.
Composition peaked at 287 MB, above the old cap. The user's session had
6.9 GB of video budget with 1.9 GB used.

**Unrelated, found while testing.** `test_native_visual_cadence` no longer
compiles. Its stub lacks `frame_ends_before_adoption` and other symbols that
`c3x_renderer.cpp` now has; it is a stale test, not a renderer failure.

## 81. Terrain keeps its shape at the fog; the fog fades like Civ III's (October 17)

**Report.** Hills and mountains changed shape as tiles were explored. Ground
next to unexplored tiles dropped away, and rose again once those tiles were
revealed.

**Cause.** `ViewerTopology` gave unexplored tiles a flat category and
`hidden_bit`, and `hidden_taper` sloped revealed ground to the 2.5 datum
within a quarter tile of them (review 2026-10-04, "gray band"). Revealing a
tile removed the taper, so its explored neighbours reshaped.

**Change: real relief** (the default; the user chose real heights).
- Unexplored tiles keep their real category, rivers and snow, and lose only
  their effect and eruption bits.
- They carry `unexplored_bit` (bit 28) instead of `hidden_bit`, so there is
  no taper. `Tile::unexplored` exposes the bit.
- `C3X_RENDERER_HIDDEN_RELIEF_CONTROL=1` restores the flat mask and taper.

**Consequence and fix.** The old gray band came back: raised front edges
(screen-down, x=column+1 and y=row) facing an unexplored tile stood above that
tile's flat fog diamond. Nothing is drawn there, so the flat underlay showed
inside the fade, as a dull band with a hard top edge.
- `terrain_mesh_body.h` now continues the surface down from the edge's own
  vertices, with their materials, to the datum a quarter tile into the
  unexplored tile.
- Revealing the tile recompiles its neighbour without the extension.

**Lesson.** Renderer64 compiles its shaders from
`packs/Renderer64ResidentRuntime`, not from the checkout. A first attempt,
a black apron selected by a new `terrain.hlsl` branch, was clipped in game as
zero coast coverage. The ground continuation needs no shader change. Any
shader edit needs a pack overlay tool (`tools/overlay_*_shading.py`).

**Fog look: native Civ III.** From the user's reference, measured against
its tile grid:
- Unexplored black and fog of war blend across the tile edge: half at the
  edge, complete a fifth (unexplored, `shroud`) or a quarter (fog of war,
  `feather`) of a tile into either tile.
- Edges are straight along tiles and rounded at corners: the bilinear blend
  the fog of war already used.
- So native shows a sliver of an unexplored tile's own ground at its rim.

Previously unexplored tiles stayed opaque and the whole fade lay on the
explored side, because nothing is drawn under them. Explored tiles now add
flat ground at the datum a quarter tile into every unexplored edge and corner
neighbour, with no relief, trees or features:
- natural terrain (`terrain_mesh_body.h`), with that rim's own materials;
- underlay, bed, water and river (`compile_ground_surfaces`; `prepare_ground`
  sets `GroundCompileInput::rim` from observed neighbours, so a reveal
  recompiles the tile).

CPU and GPU coverage match (oracle, max error 1).

**Rejected.**
- Hard per-tile states. The user asked for them while the band was taken for
  fog, then asked for native fades.
- A map-anchored wobble, and an explored-side distance fade. Native edges
  are straight and centred on the tile edge.

**Tests.**
- `test_viewer_topology`: real relief.
- `test_unexplored_apron`: rims in both compilers and the front-edge
  continuation; fails without them.
- `test_unseen_boundary`: the native blend.

**Checked.** Reveal-scroll on the new-game save, same camera before and
after: the band in the notch is gone. Unit shadows are unchanged: the
darkest shadow pixels are 0.71 and 0.69 of the ground in both runs. The grid
seen in one run had been switched on during that run.

**Unrelated, found while testing.** Two tests no longer compile, with or
without this change: `test_visibility`'s fresh-map test (missing resident-world
symbols since 934de059) and `test_fresh_preparation_cancellation`.

## 82. The fog is only an overlay, with Civ III's fade (October 17)

**Report.** The user's screenshots after review 81 showed four problems:
- Edges next to unexplored tiles fell away, then looked flat once explored.
- Hills and mountains at those edges had grainy, striped shadows, which
  disappeared when the neighbours were revealed.
- Lines cut through city building shadows.
- Fades showed a semi-opaque band where Civ III fades gradually.

The user's rule: like Civ III, the map is drawn as it really is and fog of war
and unexplored black are laid over it. The overlays have no effect on the
look of the tiles under them.

A first fix drew hidden neighbours from their explored neighbours. The user
then found reveals slow and play slower: a unit arriving next to unexplored
tiles must reveal them at once, after every step of a multi-move unit.

**Causes.**
- Review 81's apron and flat rim:
  - Raised edges facing unexplored tiles dropped a curtain to the 2.5 datum.
  - Every explored tile added flat datum ground a quarter tile into its
    unexplored neighbours.
  - Seen through the half-black fade, both read as ground falling away.
- The apron copied its edge vertices' normals onto a steep surface. Shadow
  receivers offset and bias along the normal, so it self-shadowed in stripes
  and speckles (acne). Revealing the tile removed it.
- The fade met at each tile edge as two half smoothsteps with zero slope: a
  flat, half-covered shelf. Fog of war was a 50% grey veil.
- Renderer64's shadow atlas (`SandboxSceneShadow`) compiled
  `environment_refresh/source_caster.hlsl`, which has no city branch:
  - Masked city materials (100 plus channels) fell into its natural-opacity
    branch and were clipped by whatever tree mask t33 last held. With no mask
    bound, they cast nothing.
  - The production `SourceShadow` already selects the city-profile caster.

**Civ III's fade.** `Art/Terrain/FogOfWar.pcx` holds 9×9 sprites of 128×64,
one per state of the four tiles around a tile vertex. They are drawn on the
dual grid, whose corners are tile centres:
- The column gives the top and left tiles, the row the right and bottom.
- Grey is a light multiplier: unexplored 0, fog of war 0.6, clear 1.
- The user's annotated screenshot matches the unexplored-to-fog sprite within
  a few percent.

**Reveal latency causes** (unit-motion game test, trace level 2):
- Exploration changed terrain inputs, so a reveal recompiled ground around
  it, and the fog stayed until a new view was published. A further move
  cancelled that job, so multi-move units revealed after the last step.
- `UnitArrivalVisibility` held any sight the drawn view lacked, including
  fog of war lifting over explored ground: 13 frames kept one fogged tile
  dark until the job finished, 0.8 s after arrival.
- The fog pass only admits Civ III's new sight when a job starts. A slow
  arrival frame (a 300k-pixel static repair) delays that publication by
  about 0.1 s. Not changed here.

**Changes.**
- `VisibilityCoverage::light` and `gpu_visibility.h`:
  - Bilinear seen and explored flags between tile centres, each through one
    smoothstep over 0.55 tile.
  - Seen light reaches 0.225 tile into darker tiles. Explored meets
    unexplored 0.05 tile inside the unexplored tile.
  - Fog keeps 0.6 of the light, with no grey veil.
  - Within 0.043 of the sprites.
- The hidden ring (`CapturedScene`): unexplored tiles within two tiles of
  explored land.
  - Their world inputs carry `EXPLORED` and `hidden_ring`. They compile and
    draw as ordinary terrain, without objects or borders (`window_tile`
    takes them from the input).
  - Two tiles, because a unit on a mountain sees two, and a quick second
    step can arrive before the first step's view is drawn.
  - `ViewerTopology` no longer masks terrain by exploration, so a reveal
    changes no terrain input. The apron, rim and drawer-neighbour code of
    the first fix are reverted.
  - Only live state is withheld: unexplored tiles lose the effect and
    eruption bits (24, 27; snow, 26, stays), so a ring volcano does not glow
    or bloom past the overlay. The mask reads native exploration, not the
    ring's `EXPLORED`. (Raised by the volcano session.)
- `UnitArrivalVisibility`: explored and ring tiles lift on the drawn view as
  soon as native sight arrives. Unit arrival waits and immediate loss are
  unchanged.
- Ring identity stays stable, or every job and every shadow proof rebuilt it:
  - It is the input's content revision (`appearance_revision`,
    `retained_raster_appearance`), whatever Civ III captured of the
    unexplored tile and whether or not the frame observes it.
  - Captures of unexplored tiles do not clear compiled ground, and window
    ring tiles do not rewrite the record.
  - Ring tiles seen only in the halo keep their compiled ground (`attach`).
- `fresh_pipeline.h`: the city-profile caster.
- `biq_preview.cpp`: unexplored tiles are the fog pass's, and
  `C3X_RENDERER_PREVIEW_VISIBILITY=<n>` sets the explored radius for Lab fog
  studies.
- `resident_scene.cpp`: `reveal-held` (trace level 2) counts cells kept dark
  for a unit's arrival or for a view drawing them.

**Tests.** Each was confirmed to fail on the previous code.
- `test_unexplored_ground`:
  - A reveal keeps content and extends the ring (24 tiles around one).
  - Fog of war and ring reveals lift on the drawn view; a tile three away
    waits.
  - Ring ground prepared at load survives Civ III's capture, window frames
    and unobserved frames, with one identity.
  - Halo-only ring tiles keep their compiled ground. The window draws the
    ring.
- `test_viewer_topology`: exploration never changes terrain, except that an
  unexplored volcano's activity is withheld until it is revealed.
- `test_visibility`: its stub arrival gate gains the `reveal-held` counters.
- `test_city_shadow_caster` (D3D): the selected caster clips masked city
  quads by their own mask and ignores tree masks. Old: 0 of 4096 covered.
- `test_unseen_boundary`: the fade matches `FogOfWar.pcx` curves within 0.05
  and keeps its slope at the tile edge (old: 0.3; new: 0.9–2.6).
- `test_visibility`: fog value. The GPU oracle matches the CPU (max channel
  error 1).
- Ring counts updated in `test_world_readiness`, `test_world_preparation_schedule`
  and `test_world_input_lease`.

**Measured.** Renderer64 trio built from HEAD plus this change plus the
tactical session's five files, `user-3350BC`, `-ProfileRenderer`.

| Fog lifts after visual arrival | Before | After |
|---|---|---|
| unit-turn, first move | 736 ms | 91–96 ms |
| unit-turn, second unit | 115 ms | 52–57 ms |
| unit-motion | 301 ms | 78 ms |

- Scroll, two runs each: median 59.4–59.5 fps and p10 56.5–58.0 for both;
  helper CPU 24–25%; 6 shadow builds; same proof rejections. Helper memory
  is 35–50 MiB higher.
- The first view after loading takes 1.2 s instead of 0.9 s: its job builds
  the 145 ring tiles on screen, which loading does not prepare.

**Open.**
- Six tests that copy code out of `c3x_renderer.cpp`, `fresh_pipeline.h` or
  `resident_scene.cpp` fail on HEAD too (stale extracts):
  `test_canonical_membership`, `test_fresh_shared_submission`,
  `test_fresh_preparation_cancellation`, `test_native_visual_cadence`,
  `test_visibility` (`fresh_map_without_resources`) and
  `test_zoomed_step_canonical`.
- City building shadows still show a one-pixel checkerboard in game. It is
  finer than a shadow texel, so it is not caster holes. The cause is not yet
  found: no alpha-to-coverage or screen dither is in the city path.
- A city's border edges and colour, captured for the first time near a
  moving unit, rebuild its 21 tiles' terrain (about 0.4 s of job time) even
  though borders are drawn from the occurrence. This predates this review.
  Fixed in review 83.

## 83. Borders no longer rebuild the terrain under them (October 17)

**Cause.** Border edges and colour (`territory_edge_mask`,
`territory_color_rgb`) were part of a tile's compiled identity. A border-only
change replaced the record's appearance, cleared its compiled terrain and
advanced its revision, which is in the compile context and in neighbouring
canopy proofs. Borders are never compiled: draw records and window tiles take
them from the occurrence.

**Why the first move.** `read_custom_renderer_tile` (injected) fills borders
only when `is->custom_renderer_unit_images` is set. Only the
`Units_Image_Data` trampolines set it, and `Animator::update` is skipped while
the resident front is ready. So from loading until the first unit animation,
every capture has no borders. In game, Washington shows no border before
the move and does after (window samples). Civ III reads the palette from its
`units_image_data` global. Showing borders at load needs that global, or
another palette source, in the injected capture: not changed here.

**Change.** `CapturedScene::same_compiled` compares content except borders.
- `publish()`/`update()`: a border-only change replaces the appearance and
  keeps the revision and compiled views. The tile is still published as
  changed, `appearance_epoch` advances and its raster appearance is touched.
- `matching_permitted_revision`, `appearance_revision`, `attach` and
  `update_world_input` compare with it, so a capture ahead of its publication
  and a reveal inside foreign borders keep their compiled identity.
- Borders still redraw. Membership and `terrain_content_key` include them, so
  the tile is placed again from its binding, and its records, resident-world
  copy and static-raster contributor keys carry the new edges.

**Tests.** `test_territory_identity` (three cases) fails on HEAD `4c291752`
and passes. `test_captured_scene` now expects borders to keep the identity.
The 29 modules that use `CapturedScene` or territory pass (220 tests).

**Measured.** unit-motion on `user-3350BC`, `-ProfileRenderer`, one run per
side. Isolated trios: HEAD and HEAD plus this change. Only the first (north)
move was admitted on both sides, so the scenario's two-move check failed on both.

| First move's job | HEAD | Change |
|---|---|---|
| `tile-residency-reasons` | missing_binding=21 dependency_mismatch=10 | 0 and 0 |
| built / reused | 31 / 313 | 0 / 344 |
| invalid appearance dependencies | 24 | 0 |
| setup phases (assembly, draw) | 619 ms (414, 192) | 257 ms (176, 70) |

The loading view is unchanged (145 ring tiles built on both). After the
move, the window samples show the same border on both sides.

## 84. Revealed tiles show their contents with them (October 17)

**Report.** After review 82, resources and forests appeared a moment after
the tile they were on was revealed.

**Cause.**
- The hidden ring drew unexplored tiles from object-free inputs. A reveal
  then changed the tile's content and identity.
- The next view rebuilt it and the canopy around it. Forest clearings depend
  on the routes, resources and sites of a tile and its neighbours.
- Before the fix, the next job built 23 tiles, with 47 appearance
  dependencies invalid, and the resources appeared about one sample later.

**Changes.**
- Civ III draws every tile and lays its fog over it. A ring tile it captured
  on screen now draws that capture (`CapturedScene::ring_content`):
  resources, routes and improvements as the player knows them, forests and
  borders.
  - City bodies (known only once explored) and live tile effects are
    withheld.
  - Its identity is the record's revision when nothing was withheld, so a
    reveal keeps its compiled ground and that of its neighbours.
- `UnitArrivalVisibility` remembers what a tile shows when it is revealed.
  If that differs from what the drawn view drew there (a city, or
  improvements learned with the reveal), the fog waits for a view drawing
  it, so the tile appears whole.
- This builds on review 83's `same_compiled`.

**Tests.** Each was confirmed to fail on the previous code.
- `test_unexplored_ground`:
  - The ring draws captured contents, a reveal keeps content, identity and
    compiled ground, and a city is withheld until explored.
  - A reveal with new contents waits for its view.

**Measured.** `reveal-scroll` on `newgame-4000BC`, isolated trios from HEAD
plus review 83, without and with this change.

| Scout's move | Without | With |
|---|---|---|
| fog lifts after visual arrival | 160 ms | 50 ms |
| tiles the next job built | 23 (9 dependency) | 7 (new ring tiles only) |
| invalid appearance dependencies | 47 | 0 |

- Window samples of the worker's second step:
  - Without the change, the terrain appears first, and a gold resource and
    a tribal village appear one sample later.
  - With it, the gold appears with its tile. The village tile (a hut is
    known only once seen) stays dark one sample, then appears whole.

## 85. The first view after loading keeps the prepared hidden ring (October 17)

**Report.** After review 82 the first view after loading took about 1.2 s
instead of 0.9 s. Its job rebuilt the hidden-ring tiles on screen and in the
window margin (`missing_binding=145`, `user-3350BC` scroll).

**Cause.**
- Loading does prepare the ring. Its GPU sweep admitted 688 keys: two for
  each of the 199 explored and 145 ring tiles in view.
- `WorldInputCapture::accept` rebuilt each unexplored loading-page tile from
  the packed topology alone. It dropped the map seed (`variant_seed`) and
  the pine flag, which the injected producer supplies for unexplored tiles.
- Civ III's first on-screen capture carries both. Every ring tile on screen
  changed content, so it got a new identity and `update_world_input`
  cleared its compiled ground. The first job rebuilt all of them.
  - A diagnostic build confirmed it: at that publication 145 of 145 bound
    ring tiles lost their binding, and each differed only in `variant_seed`.
  - Loading had also compiled ring pine forests as broadleaf.
- Ring tiles with canopy depend on tiles up to two away, which include
  unexplored tiles outside the ring. Their identity was 0 until Civ III
  captured them on screen, then that capture's revision. In game, 2 ring
  tiles were still rebuilt once the seed was kept.
- `resident_world_load` admitted only natively explored records, from their
  native appearance, so ring tiles were never stored at load.

**Changes.**
- `world_input_capture.h`: unexplored loading tiles keep the producer's
  seed and terrain feature flags. They are the terrain recipe, not hidden
  art.
- `CapturedScene`: an unexplored tile outside the ring takes its world
  input's content revision as its identity (`retained_raster_appearance`,
  `appearance_revision`), as ring tiles do. It is never drawn, and ring
  compilers read only its input.
- `resident_world_load` stores ring tiles from their world input, as
  `window_tile` draws them. Its trace reports `ring=`.
- `resident_water_visible`: stored ring water moves like the window's.
  Camera steps draw from the resident world, whose water took its
  visibility from native exploration, so ring water was fog-frozen there
  (no time, drift or water-scene contribution) but animated in the window.
  Storing the ring at load exposed this on every early scroll step. 10 Hz
  frames showed no visible difference: the ring lies almost entirely under
  the black.
- Not this: the user's coastal ghost after a scroll step. For one or two
  frames the shallow-water tint ends at a copy of the coastline about one
  step to the side, then corrects. It appears with and without the water
  fix and with no ring at all. Those frames are deferred steps that reuse
  the previous view's static layer shifted (`static-compose camera=-2400,-300
  slot=-2208,-300 shifted=1`) while water is redrawn each frame. The cause
  was the reused water mirror, not the static layer (review 86).

**Tests.** Each fails on the previous code.
- `test_loading_ring`:
  - A ring tile prepared from a loading page survives Civ III's capture
    with its compiled ground and identity. Old: seed 0, compiled ground
    cleared, identity 24 -> 2401.
  - An unexplored neighbour's identity is its world input's revision before
    and after Civ III's capture. Old: 0 -> 601.
  - Stored ring water is visible where the window draws it moving (the real
    `resident_water_visible`). Old: ring tile stored frozen.
- `test_world_input_lease`: an unexplored halo tile outside the ring now
  reports its input's revision instead of 0, and its reveal still changes it.

**Measured.** `user-3350BC`, `scroll`, `-ProfileRenderer -MeasureCadence`.
One isolated build with a measurement-only switch (not in the checkout):
the old stripping and identity, and no hidden ring at all. Times are from
the first `world-window` trace. A discarded warm-up run preceded each series.

| | camera-complete ticket=1 | first image on screen | ring tiles built |
|---|---|---|---|
| staged trio (old) | 1261 ms | 1971 ms | 156 |
| old behaviour, same build | 1248, 1175 ms | 1952, 1795 ms | 156 |
| this change | 1209, 1159, 1167, 1158 ms | 1834, 1645, 1905, 1659 ms | 0 |
| no hidden ring, same build | 795, 837 ms | 1523, 1530 ms | 0 |

- "First image on screen" is the first `direct-visual drawn=1`.
- The first job now builds nothing (`built=0 reused=344`), and
  `resident-world-load` stores the ring (`loaded=344 ring=145`).
- The rebuild itself cost only about 40 ms of job time. The first image
  arrives about 145 ms earlier on average than with the old behaviour.
- About 235 ms remains against no ring. It is GPU work for drawing the ring
  in the first frame: 235 water draws instead of 64, 2034 shadow projections
  instead of 1731, and 437 static raster cells instead of 275.
  - This VM's GPU runs in order. With the ring, the first frame's water
    uploads wait for that work inside the job (`water` 267-283 ms).
  - Without it, the same wait lands in the next frame (`frame-preparation-
    ready render_ms` 281-307). So camera-complete overstates the gap: the
    second frame is at 1108-1125 ms without the ring and 1172-1180 ms with
    it.
- Frame rate is unchanged. Drawn frames per second in the first 3 s:
  33.7-40.3 with this change, 33.0-38.3 with the old behaviour and 34.7-36.3
  with no ring. Median 58.5-58.9 fps for all.

**Open.**
- Later eras: Civ III's capture sets `route_style` to the viewer's era on
  every tile, while loading pages give unexplored tiles 0. From the Middle
  Ages on, the first capture again changes ring content. Ancient saves, such
  as this one, are unaffected. `route_style` also selects mine and farm art,
  so it cannot simply be ignored on tiles without routes.
- The remaining first-frame cost is the price of drawing the whole map.
  Drawing the outer ring only from the second view would move that GPU work
  later, and a reveal in the first seconds would wait for the second view.

## 86. Coastal ghost after a scroll step: the reused mirror's offset (October 17)

**Report.** For one or two frames after a scroll step, the shallows along a
coast take a greener tint that ends at a sharp, ragged copy of the coastline
out in open water, then correct (user; `user-3350BC` scroll, review 85).

**Cause.**
- Within 150 ms of a camera move, a mirror changed only by the camera is
  reused with a sampling offset instead of redrawn (review, October 3). The
  water pass reads it at screen + `NativeReflectionTarget.zw`, which was
  4 + (camera − mirror camera) × zoom.
- A pixel lands at vertex + (camera + 4) on screen and at vertex + (mirror
  camera + 8) in the mirror, both scaled about the view centre, so the offset
  is 4 + (mirror camera − camera) × zoom. The sign was reversed: the mirror
  was read two steps to the side.
- The sea mirrors terrain only where it is shallow, with full coverage
  wherever mirrored land has colour. The shallows therefore showed mirrored
  land out to the displaced coastline, with a hard, stair-stepped edge (the
  mirror's reduced raster). Rivers read the mirror the same way.
- Confirmed in the 10 Hz frames and trace: the ghost frame was shown between
  frames that reused the mirror (`reflection_draws` unchanged at the camera
  change) and the exact redraw at 227 ms; the next frame was correct. The
  static layer and water depth were registered correctly.

**Change.** `fresh_pipeline.h`: `reused_mirror_shift` gives the reused
mirror's offset with the correct sign. Reuse, its 160×96 bound and the
150 ms redraw are unchanged.

**Test.** `test_mirror_reuse_shift` projects world points through
`SceneProjection` for the screen pass (camera + 4, guard 4) and the mirror
pass (camera + 8, guard 8) at seven zooms, eight steps and nine cameras, and
requires the water pass to read the same point from a reused mirror. It also
pins the translations, guards and sampling expressions it models. Old: at
0.5×, a 64 px step read 64 display px to the side.

**Measured.** `user-3350BC`, `scroll`, `-SampleHz 10 -ProfileRenderer`; the
staged build against the same working tree plus this change, staged
temporarily. Mirrored-land band: pixels (half scale) whose green-minus-blue
changed by more than 6 between a step's first frame and the next, against the
next same-camera pair.

| | steps with a band | band, median / max | control, median / max |
|---|---|---|---|
| before | 8 of 20 | 193 / 8852 | 19 / 337 |
| after | 0 of 20 | 30 / 363 | 103 / 375 |

- "A band" is more than three times the control plus 200. The after
  control leaves out one pair that spanned the next step.

- Both builds reused the mirror on 14 of 20 moved frames and redrew it 19
  and 20 times; 1057 and 1065 traced scene frames. The change does no
  additional work.

## 87. The busy save's memory peak: attribution and three cuts (October 18)

**Why.** Music blips are whole-VM pauses of about 280 ms or more, triggered
by macOS memory pressure (`docs/audio_skips.md`). On the busy save's `near`
run the helper peaked at 5.7 GB private and 3.4 GB GPU memory, against the
G1 direction of about 4 GB and 2 GB.

**How the VM counts memory.** A D3D11 probe in the guest
(`native/build/mem-peak/d3d_*_probe.cpp`, not tracked) showed:
- Every D3D11 allocation costs exactly its size in the process's private
  bytes, with no hidden CPU copy. The helper's private memory is therefore its
  whole footprint, GPU included.
- `D3D11_USAGE_DEFAULT` buffers, such as pulled pages, count in private
  memory but not in the adapter's `CurrentUsage`.
- The driver keeps up to about 256 MB of freed allocations for reuse.

**Timeline** (t from Civ III start):
- t 52-71: the whole world goes to the GPU (+1.27 GB).
- t 76: first view, at 5.0 GB.
- t 108 and 112: minimap jumps.
- t 116-129: zoom to 2× and 3×, the peak.

macOS went to critical pressure at t 72, during the load and first view. The
Mac already held 14-16 GB in its compressor before launch, with Parallels
resident at 8-11 GB.

**Attribution at the peak.** From region walks of the helper's committed
memory by allocation size, the census and a new CPU census line:

| Item | MB | Note |
| --- | --- | --- |
| World geometry | 1,269 | resident from loading |
| Textures | about 600-800 | the census counts 261; natural and city pack textures are outside it |
| Retained DDS file bytes (CPU) | 591 | kept beside the uploaded textures for device recovery (`geometry-census-cpu dds_kb`) |
| World RAM copy (CPU) | 618 | compressed copy of the world (1.71 GB raw) |
| Static layer | 240 at 1×, 554 zoomed | 52.4 MB per slot and per overlay, 2 previews, cache 31 |
| Retained composition | 280-356 | 35-61 window-sized textures live in all |
| Pulled pages | 107, growing to 305 | `DEFAULT` buffers kept until a 512 MiB budget |
| Shadow / scene targets / units | 100 / 92 / 94 | |
| Owner metadata (CPU) | about 80 | records, chunks and proofs of 16,900 owners |

**Eviction churn.** The world-geometry budget follows free memory. On the
14 GB VM, guest free memory sat at 3.1-3.8 GB, against a reserve of
max(2 GB, RAM/6) + about 0.8 GB = 3.1 GB. Each camera frame then evicted
about 9 world tiles and restored others from the RAM copy: 3,692 restores in
one run (`base1`). The census run evicted 4,235 owners (126 MB of GPU
geometry, plus their driver blocks and heap), and `base1`'s private memory
fell by about 0.8 GB late in the run. That eviction, not a smaller working
set, is why its settled figure (4.6 GB) was lower than its peak.

**Changes** (all picture-identical):
- `c3x_renderer.cpp`: the loading sweep releases the RAM copy of every world
  key it made GPU-resident: 15,200-15,464 records, 609-612 MB. Refused keys
  (1,436-1,700, 6-9 MB) keep theirs. Nothing is released if the refused keys
  exceed the 384 MiB view allowance. Free memory then no longer evicts the
  resident world; the fixed tier (canonical + 384 MiB) alone bounds it.
  Section 88 replaced this rule: the tier is now sized by the adapter.
  `C3X_RENDERER_WORLD_BACKING_CONTROL=1` keeps the RAM copy.
- `pulled_mesh_pages.h`: pages no frame has drawn for 300 frames retire once
  residency passes 128 MiB. Retirement chooses its pages first and then erases
  their entries in one pass. A first build scanned every entry once per
  retired page, and after a zoom-out the 2× and 3× pages aged out together:
  one run stalled at 20 fps for three seconds of 1× idle.
- `fresh_pipeline.h`: while zoomed, the 1× lane's back slot and overlay
  (105 MB) are freed. Only the 1× front feeds the previews; the next 1×
  refinement or recentre allocates the slot again.
- The memory census adds `geometry-census-cpu` (owner CPU bytes and retained
  DDS bytes). `tools/memory_peak_report.py` checks a run's peak and
  world-tile churn. `sample_memory.ps1` adds a `utc_ms` column.

**Tests.**
- `test_loading_world_preparation`: release after the sweep, refused keys
  kept, control switch, and an over-allowance sweep keeps every copy.
- `test_required_geometry_admission`: the released tier ignores free memory.
- `test_pulled_mesh_pages`: idle pages retire, drawn pages stay, and 37 idle
  pages retire in one pass down to the floor (VM GPU).
- `test_static_zoom_lane_release`: only the idle 1× back slot is freed.
- `test_geometry_census`: the CPU line counts shared objects once.

Each failed against the old code or a mutated copy. `memory_peak_report.py`
fails on both pre-change captures (5.7 GB; 703 and 3,692 restores).

**Measured** (busy save, `near`, one run each, MB; GPU is the
`Total Committed` counter):

| Run | Peak private | Peak GPU | First view | 2× (t 120) | Late private / GPU | Tiles restored or compiled after loading |
| --- | --- | --- | --- | --- | --- | --- |
| Before (`base1`) | 5,707 | 3,456 | 5,065 | 5,528 | 4,629 / 2,522 | 3,692 |
| After (`fix2`) | 5,139 | 3,367 | 4,569 | 5,115 | 4,761 / 3,035 | 0 |
| Before, census (`census1`) | 5,708 | 3,483 | | | | 703 |
| After, census (`fix2c`) | 5,072 | 3,356 | | | | 0 |

- The peak fell by 570-640 MB and the first view by about 500 MB.
- The late figure is higher than before (+130 MB private, +500 MB GPU)
  because the world is no longer evicted and restored.
- Pulled pages stay at 127-211 MB instead of growing to 305. The zoomed
  static layer is 450 MB instead of 554.
- macOS still reached critical pressure, at t 91 instead of 72.
- No camera, GPU or admission failures in any run.

**Speed A/B** (user rule: memory cuts must not slow anything). HEAD's trio
(A) against this build (B) was staged in turn. Runs used `-MeasureCadence`,
alternating A B B A; medians of `near_report.py`, `zoom_report.py` and
`check_navigation_cadence.py`.

| Busy `near` (6 A, 4 B runs) | A | B |
| --- | --- | --- |
| 1× idle, 3× idle, 1× idle at the end | 47, 59, 46 fps | 49, 59, 47 fps |
| 1× scroll x: step / adoption p50 | 237 / 212 ms | 211 / 190 ms |
| 1× scroll y: step / adoption p50 | 157 / 135 ms | 157 / 134 ms |
| 2× scroll: step / adoption p50, steps | 232 / 184 ms, 19 | 156 / 131 ms, 29 |
| 3× scroll: step / adoption p50, steps | 156 / 126 ms, 30 | 120 / 93 ms, 36 |
| Zoom in to 2×, to 3×, out to 1× | 34, 58, 28 fps | 33, 55, 31 fps |
| Jump 1, jump 2 | 314, 802 ms | 446, 620 ms |

- One B run (B2) was slow in every segment, including the 1× scrolls before
  any zoom (4.3 fps at 1× y, against 30-37 in every other run). A runs had
  the same kind of disturbance (A4 1× idle 20.7 fps; A3 at 3 fps for a
  second). The medians include it.
- Jump 1 has only two B detections, one of them B2; none of the changes acts
  before the first zoom.

Profiled pair (trace level 2, `step_report.py`):
- A: 58 camera jobs, 11 over 300 ms, 12 with over 100 ms of mesh
  preparation (evicted world tiles restored); job p50 69 ms, adoption p50
  198 ms.
- B: 96 jobs, none over 300 ms, one mesh preparation; job p50 41 ms,
  adoption p50 135 ms.
- `check_camera_jobs`: no failures on either side; A needed the
  required-geometry overflow 8 times, B none. `check_native_call_waits`
  passes on all 12 busy runs.

Busy `zoom-out` (2 and 2):
- Settle mean 698 → 662 ms; longest presentation gap 155 → 117 ms; fps
  equal.
- `check_navigation_cadence` fails on both A runs (no progress for 1.13 and
  1.04 s) and passes on both B runs (0.98, 0.87 s).
- macOS was at critical pressure for 53-54 two-second samples per A run and
  38-39 per B run.

Light save `near` (2 and 2): equal in every segment (idle 60 fps; scroll
steps 78 ms). Jumps 33 → 82 and 95 → 43 ms; none of the changes acts on a
38-tile world.

**What the Mac sees** (`footprint -p` of `prl_vm_app` every 5 s, busy
`near`; idea from the music-skip session):
- The guest's 14 GB is fully dirty from boot and does not change.
- `IOAccelerator (graphics)` plus unmapped graphics is the Mac-side Metal
  mirror of the VM's D3D resources. It tracks the helper's GPU memory almost
  1:1: 0.54 GB with no game, 3.5-3.9 GB at the first view, peak 4.19 GB.
- The total footprint grows from 15.4 to 19.5 GB.

| Mac graphics | Peak | Late (t 180-245) |
| --- | --- | --- |
| HEAD trio | 4,192 MB | 3,340 MB |
| This build | 4,088 MB | 3,705 MB |

- GPU memory therefore costs the Mac twice (guest RAM and Metal), while
  guest-private savings help it only through the compressibility of freed,
  zeroed guest pages.
- This build lowers the Mac's peak slightly. Late in a run it holds about
  0.37 GB more, because it keeps the whole world GPU-resident where HEAD
  evicted geometry under guest pressure (with the 1 s restore jobs above).
- GPU-side cuts (textures, geometry, static-layer formats, composition) are
  what reduce the Mac's footprint, and they count double.

**What remains** against 4.5 GB private and 2.5 GB GPU:
- *Private, about 0.6 GB over.* The 591 MB of retained DDS bytes is the
  largest exact item. Freeing them needs `reset()`, which re-uploads every
  texture from those bytes after device loss or a camera-job exception, to
  read the files again. That is a separate change with a recovery test.
- *GPU, about 0.9 GB over.* This is mostly representation:
  - world geometry (1.27 GB; mountains 400 MB and decals 199 MB at
    68-byte vertices);
  - textures, of which 25 terrain textures are RGBA16F 1024² with mips
    (267 MB);
  - the static layer's RGBA16F + R32 slots;
  - window-sized composition textures.

  Each needs a Lab comparison and the user's visual approval.
- *Host.* The Mac enters the run already compressing 14-16 GB. With a 14 GB
  VM on a 24 GB Mac, even the 4.5 GB target may not keep macOS below
  critical pressure; host load matters as much as the helper.

## 88. A 12 GB VM: the world tier follows the adapter (October 19)

**Symptom.** After the VM was lowered from 14 to 12 GB, C3X debug mode on the
busy save froze the game for a while. The map then stayed black except for
city labels and other native overlays. The build before section 87 failed
the same way, so the cause is the VM size, not those cuts.

**Cause.** Whole-world residency sized its world geometry by free physical
memory. `FrameWorkingSet::world_geometry` keeps a reserve of
max(2 GB, RAM/6) + 0.5 GB + compile lanes (about 2.8 GB). It also holds back
the attachments not yet allocated (about 1.9 GB), so it admits nothing below
about 4.7 GB free.
- *Loading.* The 12 GB guest had 4.0 GB free at the loading sweep, so the
  sweep's geometry budget was 0. All 16,900 world keys (1.21 GB) were
  deferred. The world was never GPU-resident: views compiled their tiles,
  and the RAM copy was never released.
- *Reveal.* A debug reveal changes the viewer scope
  (`resident-world-rescope`). The resident world and its RAM copy are
  cleared, and the required world pass recompiles every region into the RAM
  store before any camera job may use it. That store's capacity also came
  from free memory less the reserve, and after the reveal it was 0. Every
  recipe was refused, so the pass never completed (`backing_saved` stayed
  false). Every camera job then returned before drawing: four
  `gpu-failure phase=camera` and three `first-map-ready result=0` per run.
  An adapter-sized geometry budget alone (first candidate) failed the same
  way.

**Changes** (`FrameWorkingSet`, `c3x_renderer.cpp`):
- `world_tier`: in the geometry tier, the world's geometry budget is what is
  owned plus 80% of the adapter's measured headroom, less the attachments
  still to come. The canonical content plus the 384 MiB view allowance still
  caps it. The adapter's budget is the OS's tier for this process: dedicated
  VRAM, or shared memory on integrated GPUs and in the VM. Allocation
  pressure still holds growth at what is owned. Without the geometry tier,
  the free-memory rules are unchanged.
- `world_recipes`: in the same tier, the world's compressed recipes may grow
  by a quarter of the commit limit's headroom above 2 GiB, and never fall
  below the previous rule. The loading sweep still releases the copies of
  resident keys (section 87).
- `world_backing_released` and `tier_geometry` are removed: the adapter tier
  replaces section 87's released/unreleased split.

**Tests.** `test_required_geometry_admission` uses the measured 12 GB numbers:
- loading sweep: the old rule admits less than the world, `world_tier`
  admits all of it;
- reveal: `required_geometry` leaves no room for the revealed world, the
  tier does;
- an exhausted adapter admits only what is owned, and the ceiling caps it;
- the reveal's 618 MB of recipes: the old capacity is below them,
  `world_recipes` admits them, and with no commit headroom nothing changes.

`test_durable_preparation` checks that packets raise the tier, that no free
memory does not shrink it below what is owned, and that allocation pressure
holds it. A header copy whose `world_recipes` returns the old capacity fails
the recipe assertion; the source assertions fail on the old code.

**Debug reveal, 12 GB VM** (busy save, `debug` scenario, readiness-relative
keys):

| Build | Loading sweep | After reveal | After hide |
| --- | --- | --- | --- |
| Section 87 commit, and the build before it | budget 0, nothing resident | 4 camera GPU failures, map black | still failing |
| Adapter tier only | 1.27 GB resident | 4 camera GPU failures, map black | still failing |
| This change | 1.27 GB resident, 618 MB RAM copy released | world pass 22.2 s, map drawn | world pass 21.8 s, map drawn |

Sampled window frames show the terrain throughout the revealed scroll and
after the hide. The run's peak was 5,560 MB private and 3,535 MB GPU, during
the reveal's recompile.

**The freeze that remains.** From the rescope to the end of the required
world pass, about 37 s, the helper presents nothing: no static presentations
and no native UI presents. Civ III keeps running underneath (native map
transactions continue), and the same stall follows the hide. The rescope and
the whole-world recompile predate sections 87 and 88. Keeping the last image
and the native UI presented during that pass, or reusing geometry that does
not depend on visibility, is separate work.

**Speed A/B on the 12 GB VM** (user rule). A = section 87's commit, B = this
change. Runs used `-MeasureCadence`, alternating, 2 runs per side per save;
the tables give the mean of each side's runs from `near_report.py` and
`zoom_report.py`.

Busy `near`: both A runs failed with first-map timeouts at the 2× zoom (one
camera job took 5.9 s); both B runs passed.

| Busy `near` | A | B |
| --- | --- | --- |
| 1× idle, 3× idle, 1× idle at the end | 52, 58, 55 fps | 51, 58, 53 fps |
| 1× scroll x: step / adoption p50, steps | 233 / 189 ms, 17.5 | 233 / 191 ms, 26 |
| 1× scroll y: step / adoption p50 | 156 / 134 ms | 232 / 178 ms (B1 307 / 220, B2 157 / 136) |
| 2× scroll: step / adoption p50, steps | 499 / 565 ms, 6.5 | 156 / 131 ms, 30 |
| 3× scroll: step / adoption p50, steps | 155 / 120 ms, 19 | 118 / 94 ms, 40 |
| Zoom in to 2×, to 3×, out to 1× | 28, 54, 35 fps | 37, 55, 36 fps |
| Jump 2 | 1,188 ms | 485 ms |

Busy `zoom-out`: zoom start 110 → 98 ms, settle 613 → 600 ms, longest
presentation gap 94 → 88 ms, fps during the zoom 22 → 22. In the second after
each step settles, B presents fewer frames (26 → 23 fps). In the first
seconds at 0.5×, B idles at about 39 fps where A idles at about 46; later at
0.5×, B matches or exceeds A. A profiled pair explains it:
- At the same camera and 0.5×, B draws 14,242 records and A 12,343. At 0.62×
  they are 10,746 and 10,556; at 1× and 3× they are equal.
- Frame preparation at 0.5× takes p50 9.9 / p90 21.5 ms in B against
  9.0 / 12.5 ms in A.
- At 12 GB, A lacks part of the world at 0.5×. B draws the whole resident
  scene, as both sides did on the 14 GB VM, where section 87 measured equal
  zoom-out fps.
- Camera jobs pass on both sides (A 32, two required-geometry overflows;
  B 31, none).

Light save `near`: equal in every segment (idle 60 fps; scroll steps 78 ms,
adoption 60 ms).

`check_native_call_waits` passes on all four busy `near` runs.

**Memory, 12 GB VM** (peak private / GPU, MB):

| Run | A | B |
| --- | --- | --- |
| Busy `near` | 4,827-4,909 / 2,647-2,739 | 5,173-5,191 / 3,289-3,346 |
| Busy `zoom-out` | 4,633-4,869 / 2,612-2,821 | 5,459-5,498 / 3,536-3,571 |
| Light `near` | 2,523-2,555 / 1,638-1,671 | 2,520-2,556 / 1,638-1,645 |

- B's busy figures match section 87's 14 GB results (5,139 / 3,367). A was
  lower only because it held none of the world.
- The guest's minimum free memory was 2.3-2.7 GB with A and 1.5-1.8 GB
  with B.
- macOS stayed at normal pressure through both profiled zoom-out runs.
- The targets of section 87 (4.5 GB private, 2.5 GB GPU) are still unmet.
  What remains is unchanged from that section.

## 89. Pack texture files leave RAM after upload (October 19)

**Why.** Section 87 found 591 MB of DDS file bytes kept in the helper's RAM
beside the textures uploaded from them (`geometry-census-cpu dds_kb`). They
served one purpose: `reset()` drops every view, and the next
`ensure_scene_assets()` uploaded the textures again from those bytes. In the
VM every byte of them is private memory. Natural and city textures already
read their files again after a reset and kept nothing.

The terrain material textures dominate. Several base colors are 4096² BC3
files (21 MB each) whose GPU textures start at the 2048² level, so RAM held
four times what the GPU did.

**Change** (`c3x_renderer.cpp`):
- Each DDS loader records the pack file of the texture member it fills
  (`texture_file`): the terrain material channels, `load_material_only`,
  `load_decal_channel`, `load_dds_bytes`, vegetation, river rocks and
  improvements. Bytes held anywhere else, such as resource animations moved
  into their list, record nothing and are kept.
- After `ensure_scene_assets()` has uploaded every texture,
  `release_texture_files()` frees the bytes of recorded members and traces
  `texture-files-released`.
- `ensure_dds_texture` given released (empty) bytes with a recorded file reads
  that file again for that upload only. Absent textures keep their contract:
  optional succeeds, required fails, and a missing file fails the upload.
- The opt-in CPU terrain preview (`C3X_RENDERER_CAMERA_PREVIEW=1`, off under
  GPU presentation) samples the terrain base colors, so they stay while it is
  enabled.
- The sandbox's `SANDBOX_PREPARATION material_bytes` diagnostic now reports
  the bytes still retained.

**Tests.**
- `test_texture_file_release` compiles the renderer's own release, census
  and upload code against a stub device. It checks:
  - recorded members are freed and other bytes are kept;
  - after a reset the files are read again and uploaded byte for byte;
  - the re-read bytes are not retained;
  - absent and missing textures keep their contract;
  - the preview keeps the terrain base colors.
  Four mutants fail it: no release, no re-read, recording temporaries, and
  ignoring the preview.
- `test_scene_asset_loading` requires the release only after every scene
  asset has loaded.

**Measured** (12 GB VM, busy 1498 AD save unless noted; A = a6c74701,
B = this change; peak private / GPU in MB):

| Run | A | B |
| --- | --- | --- |
| Busy `near` (3 runs each) | 5,108-5,178 / 3,222-3,299 | 4,598-4,642 / 3,367-3,412 |
| Busy `near`, census (1 each) | 5,108 / 3,371 | 4,546 / 3,346 |
| Busy `zoom-out` (2 each) | 5,414-5,417 / 3,367-3,563 | 4,768-4,910 / 3,503-3,555 |
| Light `near` (2 each) | 2,557-2,570 / 1,645-1,673 | 1,931-1,966 / 1,640-1,647 |

- The census run traces `texture-files-released released_kb=605626
  kept_kb=0`, and `geometry-census-cpu dds_kb` falls from 605,626 to 0.
  Loading is unchanged: `load-retained-textures` 44.5 → 45.3 ms.
- Peaks fall by 470-650 MB on every save. The median of the last minute of a
  busy `near` run falls by about 560 MB (4,706-4,771 → 4,117-4,205).
- GPU memory is untouched by the change; its peaks vary by ±150 MB between
  runs on both sides.
- The guest's minimum free memory over the busy `near` runs was 1.6-2.2 GB
  with A and 2.1-2.7 GB with B (the census pair: 2.4 and 2.1 GB).
- The busy peak is now 4.55-4.64 GB, about 0.1 GB over section 87's 4.5 GB
  target. GPU memory (3.3-3.6 GB) remains the larger gap to the 2.5 GB
  target.

**Speed A/B** (user rule; `-MeasureCadence`, alternating; means of each
side's runs). After loading, the only new work is one pass over about 260
empty byte vectors whenever scene assets are ensured.

| Busy `near` (3 and 3) | A | B |
| --- | --- | --- |
| 1× idle, 3× idle, 1× idle at the end | 52, 58, 50 fps | 54, 58, 50 fps |
| 1× scroll x: step / adoption p50, steps | 234 / 205 ms, 24.3 | 249 / 211 ms, 23.0 |
| 1× scroll y: step / adoption p50 | 157 / 136 ms | 157 / 137 ms |
| 2× scroll: step / adoption p50, steps | 156 / 132 ms, 25.3 | 169 / 134 ms, 23.3 |
| 3× scroll: step / adoption p50, steps | 152 / 108 ms, 40.3 | 152 / 118 ms, 38.3 |
| Zoom in to 2×, to 3×, out to 1× | 37, 57, 32 fps | 35, 58, 31 fps |
| Jump 1, jump 2 | 206, 612 ms | 194, 544 ms |

- The B means include one slow segment each in B1 (1× x step 277 ms) and B2
  (2× step 193 ms). Without them B matches A: 1× x 229 and 241 ms, 2× 156
  and 157 ms. A1's 3× adoption (84 ms) is the low outlier of that row; per
  run the two sides overlap (A 84-123, B 104-127 ms).
- No run on either side logged a reset, GPU failure, camera error or texture
  re-read.

Busy `zoom-out` (2 and 2): zoom start 115 → 111 ms, settle 607 → 612 ms,
fps 21 → 21, longest presentation gap 97 → 103 ms, fps in the second after
settling 22 → 22. `check_navigation_cadence` passed both A runs (0.77,
0.85 s without progress) and B2 (0.85 s). B1 had one 1.23 s stall; section
87's unchanged A runs failed it too (1.13 and 1.04 s).

Light save `near` (2 and 2): equal in every segment (idle 60 fps; scroll
steps 78 ms, adoption 60-62 ms). Jumps 56 and 54 ms in A, 86 and 67 ms in B.
B2 alone took 105 and 85 ms, within the 33-95 ms these jumps varied over
section 87's unchanged runs.

**A reading error in sections 87-88.** `near_report.py` adds up the
distance between adopted camera positions, so the camera's world-wrap jump
(8,192 px on the busy save) counts as travel. A run whose last x-scroll step
lands just after the segment ends loses that jump and reports a quarter of
the "screen px/s" of a run that includes it, with the same 128 px steps.
B1's and B3's 1× x scrolls here (470 and 493 against 1,846-1,882 px/s),
B2's 2× scroll and A3's 2× and 3× scrolls differ only by that jump. Section 88's table compared these
figures; it now lists step counts. Compare steps and step/adoption times,
not px/s, until the report handles the wrap.
