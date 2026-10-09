# Camera decoupling (G1) — design

Status: draft for review, October 8, 2026. Goal G1 in
[performance_goals.md](performance_goals.md); measurements in the
[performance review](performance_review_20261007.md), sections 12–15.

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

## Stages

Each stage is measured back to back in the VM against the previous build
(`near_report.py`, `zoom_report.py`, frame-gap analysis) and gets a
regression test that fails on the old behaviour.

1. **Camera lane (A1). Done** (review, section 16). Queue wait 121 to 70 ms.
2. **Compact resident world.** The scene can be drawn at any camera near the
   current one without a camera job, within Civ VI's memory tier
   (performance goals, G1):
   - camera-dependent caches update only their new edge: the static layer
     bounded and without whole-layer re-checks or recentring, and shadow
     pages updated by adding casters;
   - the world is compact enough to stay resident (shared instanced models,
     compact terrain), so steps upload nothing;
   - units are kept resident and updated incrementally;
   - new strips are built in bounded slices between frames.

   Done when a moved camera prepares in about the time of an unchanged one
   (1 ms against 20.7 ms) and frames keep their cadence during scrolling.
   Loading tiles ahead within today's representation was considered and set
   aside: it would grow memory (review, section 19).
3. **Adopt at Civ III's pace.** Civ III adopts each step on its own tick
   instead of waiting for the job (`injected_code.c`: the step deferral in
   `move_camera` and the gate in `scroll_at_mouse`), behind a configuration
   flag. Presentation keeps the previous step until the new map frame and
   Civ III's matching front are both ready, so overlays never misalign.
4. **A rendered glide (option C).** The presented camera eases between Civ
   III's steps, and the scene is drawn at that position every frame instead
   of sliding a finished image. Civ III's map-attached overlays are offset by
   the same amount between its ticks (as zoom already places map-attached
   HUD), screen-fixed HUD stays put, and clicks map back through the offset
   (already done for the image glide).
5. **Far jumps (B).** A coarse, always-resident view of the whole map for the
   first frame, then refinement within 500 ms.
6. **Background refinement (G4).** Shadow and static quality complete in
   later frames, spread across cores.

## Decisions for the user

- **Option C, display between steps.** Accepted on October 8 as the image
  glide, on by default since October 8 (`C3X_RENDERER_GLIDE=0` turns it off;
  review, sections 17 and 24). Stage 4 replaces it with a rendered glide.
- **First frame after a far jump.** A coarse terrain view (no units, shadows
  or detail for up to 500 ms), or today's wait.
