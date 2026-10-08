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

- **Retained view refused on the busy save.** During a camera job, ambient
  frames draw the retained completed view. `retain_completed_scene` refuses
  it when its metadata charge exceeds 32 MB; frames then fall to the 125 ms
  in-job throttle (~8 fps), which matches busy-save scroll. A trace now
  reports refusals (`completed-scene-refused`) and fork cost, with a test
  override `C3X_RENDERER_RETAIN_VIEW_MIB`.
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
by the GOG installation, so players' machines carry the same layers. The
8/16-bit mitigation is the likeliest owner of a per-call loop. Changing these
layers is an environment setting for the user to decide; this pass has not
changed them.

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
