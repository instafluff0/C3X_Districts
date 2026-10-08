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
(`ViewerTopology`) and no geometry, so a reveal must build them. The real fix is
speculative preparation of the destination's sight area during travel, which
needs a post-reveal topology view for the world preparation. It has not been
started.

**Not measured.** Input-to-move latency (key or click to Civ III's move event):
the scripted harness does not timestamp posted keys. Combat clips already follow
Civ III's own cycle durations and waits, and the half-tile approach uses native
travel speed. Combat was not re-measured in this pass.
