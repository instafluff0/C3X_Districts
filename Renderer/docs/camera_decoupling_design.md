# Camera decoupling (G1) — design

Status: staged plan agreed with the user on October 8, 2026 (see "Stages").
Goal G1 in [performance_goals.md](performance_goals.md); measurements in the
[performance review](performance_review_20261007.md), sections 12–25.

## Aim

Move Renderer64 toward the Civilization VI model: the world stays resident on
the GPU, the camera is cheap to change, and background work only adds detail.
A camera request should reach the screen within one frame of the Civ III tick
that adopts it, never after a long content job.

Done when (performance goals, G1):
- Busy scroll steps reach the screen within one frame of Civ III's tick.
- Zoom responds within one frame.
- A far jump shows its first image within 100 ms and full quality within
  500 ms.

## Constraints

- **Civ III owns the camera.** It decides each step (position and 78 ms
  timing), each jump and each zoom target. Renderer64 never shows a camera
  position that Civ III did not request. (Option C below would display
  positions between two requested steps and needs the user's decision.)
- **Civ III draws its own overlays** (labels, unit status, borders, selection,
  HUD) on its own tick, and hit-tests clicks against its own camera. What is
  on screen must stay registered to both.
- **No visible quality loss.** Stand-ins (shifted, coarse or previous content)
  may show only briefly while full quality arrives, as the static layer's zoom
  preview already does.
- **Config-off and fallback.** With custom rendering off everything is vanilla.
  When resident content is missing, fall back to today's behaviour (wait for
  the job) rather than show wrong content.
- **Windows and VM.** Removing waits helps both; no VM-only paths.

## Where the time goes today (busy save, b46)

Median busy scroll step, request to adoption: 190 ms (115–900 ms).
- request to job start: 29 ms
- camera request queued behind other publications: 121 ms
- its delivery: 21 ms
- the camera job: 94 ms (shadow pages 16, scene preparation 22, static strips
  12, tile meshes 6, plus native UI service turns)
- Civ III's next native map pass: 37 ms

Scrolling runs at 168–382 px/s (native: about 1,640 px/s at 1×) and 8–15 fps.
Jumps take 1.6–2.4 s. On the light save, steps keep native pace and appear
about 45 ms after Civ III's map pass.

## Current pipeline (verified in code)

**Civ III's side (`injected_code.c`).**
- `patch_Main_Screen_Form_scroll_at_mouse` tags edge-scroll requests. Until a
  step is adopted, each 78 ms tick asks again for the displayed camera plus
  one step, so Civ III's camera never runs ahead of the renderer.
- `patch_Main_Screen_Form_move_camera` (deferrable pans) moves Civ III's
  camera, captures the native view for the renderer
  (`capture_custom_renderer_native_view`, which starts a camera job), then
  restores the displayed view. Civ III's camera state, hit testing and
  overlays stay on the displayed camera while the job runs.
- `patch_Animator_update_display` polls once per tick
  (`settle_custom_renderer_navigation`). When the job is complete, Civ III
  adopts the camera, and its map pass (`patch_Map_Renderer_m71_Draw_Tiles`)
  draws the overlays for it; `native-handoff` is traced there.

**The renderer's side.**
- `AsyncSceneClient::camera_begin` (`sandbox/async_scene_client.h`) posts
  the request to the bridge's publication queue. The latest camera wins, and
  it may pass queued `images`, `tactical` and `present` entries. But a single
  transport thread sends every entry to the helper synchronously, so a camera
  request still waits for the entry in flight. On the busy save that is often
  an image batch the helper serves slowly while a job runs.
- In the helper, the camera job prepares and renders the scene at the new
  camera, then publishes the completed map.
- Every frame, the visual cadence renders the scene from resident content at
  the completed job's camera when anything animates, and composes Civ III's
  native front over it.

A busy step is therefore serial: request, then the queue, then the full job,
then the next tick's adoption, then Civ III's map pass, then a frame. Civ III
cannot request the next step until all of that is done.

**Zoom is already partly decoupled.** The presented zoom eases over 166 ms
(`ZoomTransition`). Civ III's world image is scaled every frame
(`layers.view`), map-attached HUD items follow their anchors
(`placed_batch`), and the static layer shows a resampled preview while it
refines. Light-save zoom starts within 29–45 ms.

## Design

The key choice is whether Civ III keeps waiting for the renderer before it
adopts a camera.

**A. Keep Civ III's adoption gate; make the gate fast (recommended first).**
Civ III still adopts a step only when the renderer can show it, so its
overlays and hit testing always match the picture and nothing has to be
shifted in image space. The work is to make "the renderer can show it" take
less than one tick:
1. **A camera lane.** Camera begin, cancel and poll stop waiting for the
   synchronous image-batch call in flight: a second transport thread and IPC
   slot for camera commands, which the helper serves at its next checkpoint.
2. **A two-phase camera job.** First a presentable frame at the new camera
   from resident content: the static layer shifted within its retained slot,
   meshes only for newly exposed tiles, and the current shadow pages. Then the
   rest (shadow-page proofs, static strips at full quality) as refinement, in
   the same frames that already show the camera.
3. **Prepare ahead in the scroll direction.** While a step is pending, build
   the next strip's tiles and static content, so the following step needs
   almost nothing new.

If a step's presentable frame is ready within one tick, Civ III adopts at its
native pace (one 128 px step per 78 ms at 1×).

**B. Adopt immediately and shift overlays (needed for jumps and option C).**
Civ III adopts the new camera at once; the renderer draws it from whatever is
resident, with stand-ins where content is missing, and Civ III's overlays are
redrawn for the new camera on its tick. This removes the wait entirely but
needs a stand-in for everything not resident. It is required for the 100 ms
jump target, because a far jump has nothing resident.

**Stand-ins.** Only content that is not resident yet: newly exposed edges
(the previous neighbouring content cannot fill them, so a coarse terrain-only
version) and, after a far jump, the whole view. The static layer's existing
preview, bootstrap and refine path is the model.

## Findings from stage 1 (performance review, section 16)

- Letting camera requests run during an image wait cut their queue wait from
  121 to 70 ms p50 but left steps at 3–4 ticks. The rest of the wait is the
  previous step's synchronous adoption call and synchronous presents, which
  share the single transport thread.
- A step is adopted on the first tick after its job finishes, and the busy job
  is about 79 ms: static strip composition and the scene work before it 18,
  shadow pages for the new strip 17.5, native UI service turns inside the job
  14 (16 turns), unit snapshots 7.6, scene callbacks 6, tactical overlays 6.
  At an unchanged camera the same scene prepares in about 1 ms.
- Shadow pages already rebuild incrementally (about 8 pages per step).
  Preparing the next step ahead would need a speculative render of the
  predicted camera, which moves state the visible frames use (the static
  slot), and at best brings steps to 2 ticks.
- Civ III's captured tile set covers about 2.8 times the viewport, so the
  next step's tiles are already captured.

## Civ VI model and what differs here

Civ VI's internals are not confirmed; this is inferred from its behaviour.

| | Civ VI (inferred) | Renderer64 (October 8) |
|---|---|---|
| Camera | moved by the renderer every frame, eased | Civ III's 128 px steps per 78 ms tick |
| New camera position | a view change; the world is resident | a camera job (about 21 ms on the 3350 BC save, 79 ms busy) |
| Frames during camera work | unaffected | only from the job's checkpoints, at most 30 Hz (review, section 18) |
| World-anchored UI | re-projected every frame | Civ III's 2D front, redrawn on its tick for its camera |
| Picking | against the shown camera | Civ III's camera, corrected by the presented zoom and slide |
| Far jump | the whole map is resident | a full job at the new area (busy save 1.6–2.4 s) |

Civ III keeps choosing camera targets and step timing and keeps drawing its
overlays on its tick. The renderer side can follow the Civ VI model.

## What ties a frame to the camera job (verified in code, October 8)

Every animated frame already re-renders the scene with the camera as an input
(`retain_visual_map` → `c3x_renderer64_render_fresh`; the camera comes from the
copied frame's tile anchors), and zoom already renders at cameras no job
produced. Three things pin frames to the last camera job:

1. **Caches are keyed to residency and the camera, not to world content.**
   - The static layer's validation key includes the resident geometry revision
     and lease order (`raster_validation_key`), so every step forces a full
     membership proof. Its slot is anchored at the camera of its draw and
     recentres when a step overruns the 320 × 192 px margin.
   - Shadow pages lie on a light-space lattice, but reuse fails whenever the
     resident set changes (`prepared_signature != view_revision`), so every
     caster is re-collected and re-proven. The sampling span follows the
     receiver extent; a refit invalidates every page and the static layer.
   - The resident mesh set is the viewport plus a 2-tile ring, rebuilt by the
     job; waves cover the render tiles ±1 cell.
2. **The job and the frames share one thread** (the owner of the D3D11
   immediate context). Frames during a job run only at its checkpoints, and a
   camera-move job retires the completed view, so frames hold until it ends.
3. **Civ III's interface is replayed every frame.** Its operations (text
   curves, 5-bit blends, palette lookups, keyed shadows) read the 16-bit map
   words underneath and are reproduced bit-exactly, so every node over map
   pixels re-runs when the animated map changes.

Civ III already adopts a step on the tick after its request, as native draws
it, whenever the renderer is ready (`settle_custom_renderer_navigation`). A
camera move that is not deferred instead blocks Civ III's map pass until the
renderer's job finishes (`composite_custom_renderer_frame`, "first-map-wait").

**Where Civ VI is not followed.**
- The pre-drawn static layer stays: redrawing the static scene every frame is
  about 5,900 draws (about 130 ms in the VM).
- Civ III's interface stays bit-exact; it is not approximated with alpha
  blending.

## Stages

Each stage is measured back to back in the VM against the previous build
(`near_report.py`, `zoom_report.py`, `step_report.py`, frame gaps, the seam
check on 10 Hz window frames) and gets a regression test that fails on the old
behaviour. Expected effects are estimates from per-phase traces until measured.

- **Done before this plan:** the camera lane (review, section 16), unit types
  on demand and the zoom-lane release (sections 21, 22). The image glide
  (sections 17, 24) was removed on October 9 at the user's request: each Civ
  III camera step is shown as a jump to the camera Civ III chose (section 30).

0. **Baseline. Done** (review, sections 25–26). Tracked seam and frame-gap
   checks (`seam_report.py`, `frame_gap_report.py`; an overlay-alignment check
   comes with stage 2c); `near` twice on the busy, user and light saves and one
   busy memory run. The baseline exposed stale glide strips (fixed, then the
   glide was removed).
1. **Caches tied to the world, not the camera.** Partly done (review,
   sections 25–26): the shadow sampling span is fixed per receiver region and
   remembered per zoom level. The busy step job did not change beyond run
   noise, because residency itself follows the camera: Civ III's capture
   defines the loaded tiles around each step, so casters and contributors
   churn at every step, and a stale static layer cannot finish refining while
   the camera moves. On October 8 the user agreed to fold the rest of stage 1
   into stage 2.
2. **A renderer-owned resident world; the camera as a frame input.**
   - 2a. **Resident world. On by default since October 10** (`C3X_RENDERER_WORLD_WINDOW=0` opts out). The light-save stall that kept it opt-in was the camera delta's wrapped copies (review, sections 51–52).
     (review, sections 27–30). The renderer selects its resident set by a
     block-anchored world window, and Civ III's capture reaches past the
     view as appearance-only tiles. Steps that keep the set take 40–48 ms of
     job (p50; 25 ms for a plain 1× step); block crossings take 164 ms
     because the entering band's geometry is restored from RAM inside the
     step. Off by default until 2c moves that work off the step.
   - 2b. **World-anchored static layer.** A wrap-around slot per lane, so
     refinement and new strips continue during a scroll and nothing
     recentres.
   - 2c. **Steps without camera jobs.** A step inside the resident world is a
     frame-level camera update; the new edge's strips and shadow pages are
     filled in bounded slices; jobs remain for content changes and far
     jumps, and frames keep animating while they run. The hidden canonical
     1× lane is not redrawn on zoomed steps. Today a covered step still
     costs about 100 ms from request to adoption: about 20 ms before the
     worker starts the job (it is composing a display frame, 21 ms p50 on
     the busy save), 14 ms of synchronous delivery, the job, and the wait
     for Civ III's next poll. A block crossing costs 100–170 ms even with
     its geometry already resident: shadow proofs for the entering casters,
     resource preparation and static repair, each proportional to the band;
     warming the band in background threads cost more Civ III time in the
     VM than it saved (review, section 31).
   - Civ III then adopts each step on its next tick through the existing
     poll. Only if steps still slip: adopt without the deferral (the old
     stage 3), behind a flag.
   - Expected: busy steps on every tick (78 ms, from 230–470 ms); 3350 BC
     scroll frame-gap p90 43–47 → ≤ 20 ms. Each piece is measured on its own.
   - Verify: Civ III's draw to the presented frame ≤ one frame at p90; frame
     gaps; a step inside the resident world starts no job and changes no
     residency.
   - **Order (the user, October 9):** stage 3 comes before the rest of 2c.
     Every step first waits for the worker to finish composing Civ III's
     interface into the current frame; stage 3 removes most of that work.
   - **How a step's period forms (review, section 38).** Each Civ III tick
     adopts a ready step, redraws, then requests the next step. When that
     frame work passes 78 ms, the next tick fires before any new step can
     be ready, so the period is at least two ticks until Civ III's frame
     work is short. Measured busy steps took 3–4 ticks.
   - **2c.1, steps drawn at adoption. In place** with the window (section
     38): a step that keeps the window and its tile content completes at
     request with Civ III's tile ownership and is drawn once, by its
     adoption. Same-build A/B: median period 286 → 222 ms over the four
     scroll segments; deferred steps take two ticks (156–231 ms).
   - **2c.2, the draw outside the adoption: tried and reverted** (section
     39). Steps are bound by the helper worker's serial work per step, not
     by where the draw runs: about 55 ms per covered step, of which about
     35 ms re-prepares an unchanged scene for the new camera.
   - **2c.3, bounded finish (the user, October 9). Done** (section 41):
     - the region of interest follows window blocks;
     - the static proof carries across scroll strips;
     - the click-test backlog bound is 4096.

     Median busy step period over the four segments: 286 ms with deferral
     off, then 222 with 2c.1, now 188. Covered steps run at a steady two
     ticks.
   - **Left after 2c:**
     - block crossings, 270–570 ms on 25–30% of steps (2b's wrap-around
       slot and the entering band's work);
     - about 13 ms of setup and 5 ms of untimed static time per covered
       step;
     - Civ III's own per-tick time.

     Stage 4 is next.
3. **The interface as its own layer** (the H3 design in
   [camera_follow_and_hud_layer.md](camera_follow_and_hud_layer.md)).
   - At each front commit, the interface above the map is compiled into a
     retained layer of map-independent pixels plus a sparse program for
     map-dependent pixels, run once in the final composite against the scene
     pixel (computing its 16-bit word in place). No full-screen quantized
     copy and no per-frame replay.
   - Map-attached items stay a separate layer, shifted or placed with the
     camera.
   - Expected: light zoom toward ≥ 55 fps (G2), busy idle +5–10 fps (G3),
     shorter image queues (G5), interface memory 0.32–0.39 → about 0.1 GB.
   - Verify: bit-exact against today's interpreter over recorded native
     batches across animated frames; the existing HUD recipe oracles.
   - **Measured starting point** (review, section 32): a busy idle frame
     runs about seven full-screen GPU passes of interface replay (the
     projected view and its quantized words, two HUD base copies, a
     full-screen HUD dispatch, the world selection, front assembly, Civ
     III's full-screen keyed canvas transfer, display) and about 2 ms of
     CPU, because every operation reading the map re-runs when the map
     image changes, which is every frame.
   - **Phases**, each exact against the interpreter (`compiled_enabled`
     off) before the next:
     - 3.1 The view transform and the HUD program in one pass: the spatial
       program starts from the projected scene pixel (detail and quantized
       word computed in registers) instead of a copied world view. Removes
       the view pass and both base copies.
     - 3.2 The pointwise operations after the HUD (fixed shadows, Civ III's
       keyed canvas transfer onto the screen, button images over the map)
       join the same program; the world selection borrows its plane.
     - 3.3 The front is composited at display time (H2) instead of copied
       into a retained front texture.
     - 3.4 Pixels proven independent of the map (the HUD cache's resolved
       class, extended through 3.2) are kept across frames and zoom
       placements; only map-dependent pixels run per frame.
   - Cross-position reads (copies between positions, sprites from other
     canvases) stay in the interpreter, as today.
   - **Status (October 9):** 3.1 and 3.2 are in place as one fused pass
     for the screen-canvas transfer over the HUD over the projected view
     (review, section 35): busy idle 39.0 → 41.7 fps at 1×, 47.2 → 50.9 at
     3×, 42.0 → 47.9 in the last idle segment. Next: the remaining full-screen work per frame (interface canvas
     assembly, the scene quantization for panel blends, front assembly).
4. **Zoom drawn from the scene.** Zoom transitions sample the
   world-anchored lanes, removing the edge seams (review, section 24).
   Expected: zoom responds within one frame. (Scroll steps are not
   animated: the user retired the glide on October 9.)
   - The section 24 seams no longer appear: there were no seam frames in
     any zoom segment of the October 9 cadence runs.
   - **4.1, the wheel request reaches the next frame. Done** (review,
     section 42). The request is applied 6–25 ms after the wheel; before,
     it waited 150–250 ms in Civ III's ordered stream. The first presented
     change still takes 29–317 ms, because composing a frame at a new scale
     costs 50–125 ms in the VM.
   - **4.2, cheap transition frames.** Agreed with the user on October 9:
     make a frame at a changing zoom cheap. Stand-in frames remain only an
     opt-in fallback.
     - Attribution (review, section 43): the composition adds only 2–4 ms.
       A transition frame (~45–55 ms in the VM) is:
       - the scene re-rendered at the new scale (10–14 ms of CPU);
       - re-projection (2–7 ms);
       - display stalls on about every other frame (12–28 ms);
       - waits while Civ III's interface work holds the renderer's lock
         (up to ~26 ms).
     - Kept:
       - destination refinement waits while the zoom visibly moves
         (`zoom_moving`), which cut the p90 frame interval from 120–138 to
         84–102 ms;
       - a zoom-in keeps the wider shadow field until it settles.
     - A GPU-backlog-steered static budget was tried and reverted.
     - Further smoothness needs one of:
       - display frames that do not wait on Civ III's submissions;
       - stand-in frames (opt-in).
   - **4.3, the map HUD drawn by the renderer.** Agreed with the user on
     October 9.
     - Civ III no longer draws:
       - the unit HUD (health bar, flag, stack number);
       - city labels;
       - map messages.
     - The injected hooks skip the native draw and report only the facts.
     - The renderer draws these elements every frame, at screen resolution,
       pinned to their world anchors, with Civ III's own fonts and sprites so
       they look the same and sit in the same place.
     - Expected:
       - crisp at any zoom, during a zoom and while scrolling;
       - no per-tick native unit pass work (about 930 renderer commands a
         tick on the busy save);
       - no Civ III interface batches competing with frames;
       - the native HUD capture and re-placement machinery is retired.
     - Order: unit HUD (with the soft selection ring), then city labels, then
       map messages. Each step is compared side by side with native, tested
       and measured.
     - Done:
       - the held route and the ring at display resolution (review,
         section 46);
       - the unit status, as facts drawn by the renderer (section 47). It
         matches native, and idle ticks went from about 930 to about 240
         renderer commands.
     - Next: city labels, deferred by the user on October 10. They need new
       hook entries in `civ_prog_objects.csv`.
     - Every other vanilla interface element (minimap, panels) is unchanged.
       So is the configuration-off path.
   - **Scrolling at 2×/3× is soft** (user report, October 9; measured in
     section 43). While the camera moves, the zoomed scene is not drawn, so
     frames stretch the step's 1× image. **Fixed October 10** (review,
     sections 49–51): each adopted step is also prepared at the presented
     zoom, and that draw stays displayable during the next camera job. Soft
     scrolled frames went from 60–75% to about 1%.
5. **A compact, instanced world within a memory tier.**
   - A byte census by layer first (`world-streaming-cost`).
   - Picture-identical changes in order of bytes per risk: city building
     parts as shared instanced models (their placements are already recorded),
     cliffs instanced, ordered rigid packets no longer re-copying meshes,
     leaner vertex formats. Only if still over the tier: terrain generated on
     the GPU from the height lattice (Lab comparison).
   - The geometry budget becomes a fixed tier instead of following free
     memory.
   - Expected: busy save 3.4 GB GPU / 6.3 GB process (median) toward about
     2 / 4 GB, with the whole explored world resident.
6. **Far jumps (B).** No coarse view (the user, October 10). The whole
   explored world stays resident when it fits the memory tier, so a far jump
   lands in loaded geometry; larger maps may exceed Civ VI's memory figures.
   **Order agreed October 10:** split the helper's single worker first, then
   whole-world residency, then stage 5 compaction.

If busy 1× idle still misses 55 fps after stage 3, unit-part batching (G3)
comes next.

## Decisions for the user

- **Option C, display between steps.** Accepted on October 8 as the image
  glide and retired by the user on October 9: Civ III's scroll steps are
  shown as jumps, as in the vanilla game (review, section 30).
- **First frame after a far jump** (stage 6). Decided by the user on
  October 10: **no coarse view**; it would be noticeable and look odd. Far
  jumps are served by whole-world residency instead (the window covering the
  explored world when it fits the memory tier).
