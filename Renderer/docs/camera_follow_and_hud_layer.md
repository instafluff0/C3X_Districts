# Camera-follow scrolling and the retained HUD layer — design

**Status, October 7, 2026.** At the user's request, Civ III's own edge
scroll drives the camera again. C1 (held motion in a custom scroll timer) and
C3 (image-space slides between steps) are removed; each adopted step shows
exactly the camera Civ III chose. See
[performance review, October 7](performance_review_20261007.md).

October 5, 2026. Game Integration work following
[the performance review](performance_review_20261004.md), findings 4k and 5.
Both items change how the composition treats map-space and screen-space
pixels, so they are planned together and built in stages.

## Measured starting point (busy 1498 AD save, 0.5×)

**One edge-scroll step**

1. A timer tick moves the native camera and captures the scene (~6 ms). It
   then requests a camera job and restores the displayed camera.
2. The camera job takes 450–900 ms:
   - world preparation for entering tiles: 100–200 ms;
   - scene compose: 75–185 ms;
   - service turns: 125–540 ms.
3. The native map pass takes 230–380 ms (p50 ~280 ms unprofiled). About
   100 ms of that is ~60 city labels; the rest is native image operations and
   publications. Then the native HUD commits over the new map, and only then
   does the camera visibly move.
4. Ticks that arrive while a job is in flight are discarded, each after a full
   capture traversal (`Navigation::request` answers PENDING, then the move
   restores the displayed camera). Every completed job therefore advances
   exactly one tick of at most 50 ms of motion, about 90 screen px. That is
   ~360 native px/s, against ~3,600 configured.
5. Visual frames during a job show the frozen old camera. They run as service
   turns inside a job that mutates the geometry selection and the fresh
   pipeline's state in place.

**One animated frame**

- About nine full-screen copies (~24 M px):
  - view assembly;
  - two HUD-batch base copies;
  - two owned `select_world` planes when fixed shadows are present;
  - front fragments;
  - re-runs of map-dependent UI.
- Composition plus present limits frames to ~20 ms (~49 fps even when the
  scene is skipped).

## Principles

- Civ III stays authoritative for the camera, HUD content and picking. The
  renderer only presents a translation between native passes, and every
  translation converges exactly on the camera of the next committed native
  pass, with no visible jump.
- Pan uses the same projection as zoom:
  - the map and world overlays (`world_detail`, `projects_scene`) move with
    the scene;
  - HUD scopes (`hud_begin`/`hud_end`, `placed_batch`) move by their anchors;
  - fixed UI (`fixed_ui_*`, GUI, minimap) never moves.
- Map-dependent HUD pixels can never be cached as a premultiplied "over"
  layer. These are text curves, 5-bit blends, palette lookups and keyed
  shadows over the map. Caching is allowed only where a pixel is proven
  independent of the map.
- No new `civ_prog_objects.csv` entries. Every hook involved already exists
  and checks `enable_custom_rendering`.

## Stages

Each stage lands only after its tests pass and an unprofiled
busy-save run shows the effect. Ordering is by value against risk.

**Status (October 5):**

- **Done:**
  - C1;
  - the C2 in-job frame throttle;
  - a further C2 step, found by profiling: the camera request now overtakes
    queued native UI work in the bridge's publication queue.
- **Effect:** 0.5× busy-save scrolling went from 75 to 250 native px/s
  (review 4r).
- **Next:** H1, then C3.

**Update (later October 5):**

- **H1 was measured and is not the limiter.** On the GPU, composition
  evaluate, assemble and display take ~0.1 ms each. The display CPU time was
  a swap-chain back-buffer wait; three buffers fix it (review 4s).
- **C3 is implemented as an image-space slide in `select_world`.** It needs
  no job split or off-screen rendering. The previous world view fills the
  trailing strip, and picking subtracts the presented slide.
- **A 1× and closer benchmark (`near`) found that far jumps broke rendering.**
  The fix is the zoom-adaptive capture envelope plus an admission reclaim.
- **Still open:**
  - C4 (live rendering during camera jobs);
  - 3× water-reflection redraws;
  - the multi-notch zoom-out recapture.

### C1. Keep scroll input (small; native plus bridge)

- Add a side-effect-free navigation query, `C3X_NAV_PENDING`. While a request
  is in flight and not yet offered, the edge-scroll timer keeps accumulating
  movement in its existing remainders. It does not move, capture or discard.
- The first tick after adoption requests the displayed camera plus the
  accumulated motion. That motion is clamped to the static-raster reuse
  margin (256×160 screen px), so recentering stays a shift copy.
- The per-tick cap rises from 50 to 100 ms. Late ticks then carry more
  motion, still bounded by the clamp.
- **Expected:** 2–3× the scroll distance per second, with the same number of
  camera jobs and one capture traversal saved per dropped tick. Steps get
  larger, so motion stays stepped until C3/C4.

### H1. Remove avoidable full-screen copies (composition only)

- **Fixed notification shadows join the HUD batch.** They are placed at the
  screen centre with zero zoom offset and keep their original order after the
  HUD. `world_view_*` then stays one exact plane, and `select_world` borrows
  it instead of copying two owned planes.
- **The HUD batch stops making two base copies of the view output.** When the
  view node's single exact planes have exactly one consumer, the view writes
  into the batch pair directly. The view re-runs whenever the batch must
  re-run.
- **Before starting,** one profiled run with `C3X_RENDERER_PROFILE=1` splits
  GPU time across `compose_prepare`, `compose_evaluate`, `compose_assemble`
  and `compose_display`. That confirms the copies, not presentation, set the
  ceiling.
- **Verification:** the existing fullscreen HUD recipe oracles must stay
  pixel-exact. `copied_pixels` should fall by ~4 screens per frame.

### C3. Pan in the composition (no visible change on its own)

- Add a `pan_target` native command, carrying the requested minus displayed
  native camera. It is modelled on `zoom_target` and travels in the same
  ordered queue as `insert_map`.
- The helper owns a presented offset, a critically damped spring like
  `ZoomTransition`.
- When a map is published, the offset re-bases by that map's camera delta,
  so adoption causes no jump.
- `SceneProjection` gains a translation. `project()`, `evaluate_projected`
  and placed HUD anchors add it; fixed UI does not.
- Picking (`get_tile_coords_under_mouse`) adds the presented pan, the same
  way it already uses `ZOOM_PRESENTED`.
- DISCARD/BARRIER, exact centring, the city spotlight, combat and popups
  reset the offset.

### C2. Shorter camera jobs

- **Skip empty ambient frames.** During a job, an ambient frame that can only
  re-present the frozen map is skipped unless a UI commit is pending. Its GPU
  waits currently stretch service turns to 60–260 ms.
- **Directional prefetch.**
  - Widen the capture halo in the scroll direction, so the next job takes the
    cheap covered-membership path.
  - This is memory-guarded: the busy save sits near the VM's tile budget (4o),
    so the halo grows in one direction only and shrinks back when scrolling
    stops.
- **Measure the native pass per operation.** If city labels and native image
  operations wait on helper round trips, batch them, as was done for game
  facts (3b).

### C4. Live follow

Visual frames render the resident scene at the presented camera (published
camera plus offset) instead of shifting a frozen image.

- **What makes it possible:**
  - every camera input derives from tile anchors;
  - static rasters already reuse shifts within their margins;
  - mirror and fog are already world-anchored.
- **The job is split in two:**
  - Phase A prepares entering tiles without retiring the displayed selection.
    Visual frames stay live during it.
  - Phase B is a short commit of membership plus compose. Only phase B
    freezes frames.
- The offset is limited to the resident halo. Then the per-tick cap is
  removed and ticks are extrapolated by velocity.

### H2–H4. Later composition work

- **H2.** Composite the map fragments at display time instead of copying them
  into the front (about −1 screen).
- **H3.** Keep a retained above-map layer. Each pixel is classified, extending
  the batch's existing per-pixel classification, as static, pass-through or
  map-dependent. A frame then re-runs only map-dependent tiles and composites
  the rest once. Pan becomes a composite-time offset for map-anchored HUD.
  Any cross-position read, recompile or zoom transition falls back to full
  replay.
- **H4.** Measure first whether display CPU (~7 ms) is render-target binding
  or a driver flush.

## Measurement

| Use | Command or script |
| --- | --- |
| Busy save | `zoom-out` scenario, 180 s, `-MeasureCadence`, trace 0 |
| Light save | `forest-shadow` scenario (light save) |
| Scroll metric | `.cache/perf-review-20261004/smoothness.py` |
| Phase metric | `phases.py` on profiled runs |
| Job cycle | from `edge-scroll` and `native-handoff` (input-traced scenarios) |

Scroll is reported as native px/s over the scroll segment, the visible
step interval, and the largest presented gap. FPS is reported per segment, as
in the review's results table.
