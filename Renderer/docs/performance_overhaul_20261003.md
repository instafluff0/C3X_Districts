# Renderer64 performance overhaul — October 3, 2026

Goal: ~60 FPS during idle, scrolling, zoom and map jumps on the Windows VM,
treating memory as plentiful. This note records each major problem found in
the inherited build, the industry-standard remedy, what was implemented, the
switches that restore the old behavior, and what remains. It supersedes the
"no FPS tuning before the integration gate" guidance for the items below.

The checkout was not run here (no Windows/D3D11 host). All three build
products (x64 renderer DLL, x64 helper, x86 bridge) were syntax/semantics
checked with mingw-w64 against a baseline of the unmodified tree: the only
diagnostics are the same pre-existing GCC-vs-MSVC differences. Strict
`-Wall -Wextra -Wshadow -Wconversion` reports nothing on changed lines.

## How a frame actually spends its time (before)

| Situation | What happened | Approx. cost on the VM |
| --- | --- | --- |
| Idle, animated water/units | Retained static pixels + dynamic layers; paced by a free-running 16.667 ms timer and `Present(0)` | 16–20 ms, with periodic missed vsyncs (live ~50 FPS) |
| Every intermediate zoom value | **Full static redraw** of the visible region (each notch animates ~25 distinct scales) | ~130 ms each (~600 ms at night with cities) |
| Scroll, fractional or odd pixel shift | Full static redraw (exact-lattice rule) | ~130 ms |
| Scroll past the 320×192 px guard band | Full static redraw | ~130 ms |
| Visible receiver set changes (most scroll steps) | Shadow sampling span refit → all shadow pages redrawn → `shadow.builds++` → full static redraw | shadow pages + ~130 ms |
| Zoom ≠ 1× and any camera change | Camera transaction draws 1× and the visual frame draws at the zoom; each refits the shadow field for its own receivers, so they **invalidate each other** → two full redraws per step | ~260 ms+ |
| A city enters/leaves the visible set | City-light selection change → `invalidate_all` → full redraw | ~130 ms |
| Map jump | Full synchronous redraw inside the camera transaction; the worker is busy so ambient frames return BUSY (display freezes) | 130 ms+ freeze |

The ~130 ms "full redraw floor" is mostly per-pixel shading of the
Civ VI–derived terrain/feature/city shaders (20–35 texture samples, 9-tap
PCF, local lights) through Parallels' D3D11 translation. The single biggest
lever is therefore **not doing that work synchronously, and not doing it at
all when the pixels already exist**.

## 1. Retained static layer: preview first, refine in the background

**Problem.** The static layer was valid only for an exact (zoom, lattice
phase, guard band, shadow build, light set) tuple. Anything else forced a
synchronous full-quality redraw inside the frame (or inside the camera
transaction, freezing ambient animation).

**Industry practice.** Map/strategy renderers (and every tiled map
viewer) separate *display* from *refinement*: show the best image already
available — translated, resampled, or low resolution — at full frame rate,
and rebuild full quality incrementally under a per-frame budget, swapping it
in when complete (double-buffered progressive refinement, as in clipmaps,
virtual texturing and Google-Maps-style tile LOD). Work is time-sliced so a
frame never exceeds its vsync budget for background quality.

**Implemented** (`sandbox/fresh_pipeline.h` `compose_static`,
`sandbox/static_raster_state.h`, `native/render_core/linear_target.h`):

- **Lanes and slots.** Lane 0 is the canonical 1× projection (camera
  transactions), lane 1 any other display zoom. Each lane has a *front*
  (displayed) and *back* (being refined) region slot. A slot whose content key
  (scope, lighting, hour/season, shadow sampling identity, water class)
  changes becomes **stale but displayable** instead of discarded.
- **Whole-pixel reuse at any zoom.** Odd shifts are allowed (retained pixels
  and new strips share one region lattice; only the derivative-quad phase of
  already shaded pixels differs from a fresh render — imperceptible).
  Fractional shifts (non-endpoint zoom + camera move) round to whole pixels
  and move the whole frame's dynamic layers by the same sub-pixel snap, so
  static and dynamic stay aligned (error < 0.5 display px vs. Civ III).
- **Guard-band recentering by copy.** Scrolling past the guard band copies
  the overlapping pixels into the lane's other slot at the new anchor
  (one full-region shader copy) and draws only the newly exposed strips.
- **Zoom preview.** While the zoom animates, the front raster (at its own
  projection) is affinely resampled into the frame — bilinear color, nearest
  depth — and the dynamic layers (water, waves, units, borders) render at the
  true zoom against that depth. Missing coverage (zoom-out) falls back to the
  lane-0 1× raster, which covers every zoom ≥ 1×.
- **Prefetch at the destination.** The zoom transition publishes its
  destination (`zoom_destination_hint`); lane 1 refines its back slot at the
  destination while the animation is still running, so full quality is
  usually ready when the zoom lands.
- **Low-resolution bootstrap for jumps.** When nothing retained covers the
  new view (large jump, cold start), the region (visible area plus guard
  margins, so it survives the next scroll steps) is rasterized once at half
  resolution (¼ of the pixels) and shown immediately; full quality follows
  progressively. `C3X_RENDERER_BOOTSTRAP_SCALE` controls the scale (lower is a
  faster, softer first frame).
- **Budgeted refinement.** Back-slot bands are drawn under an adaptive
  pixels-per-frame budget (AIMD on the measured frame interval: ×0.7 above
  24 ms, ×1.2 below 18.5 ms; 150 K–32 M pixels). The slot is promoted only
  when its visible rectangle is complete, so partial refinement is never
  shown. Small scroll strips needed for the current image are still drawn
  immediately; a large gap is filled progressively behind a preview.
- **Background guard-band fill.** On frames that render anyway, a fresh
  raster grows toward its full region (margins) under half the budget, so
  scroll steps usually find their pixels already drawn; synchronous strips
  now draw only the newly visible area (no 128 px look-ahead).
- **Frames keep coming while refining.** The visual-frame scheduler now also
  renders while `c3x_renderer64_static_refinement_pending(zoom)` is true, so a
  quiet map still completes its refinement.
- The canonical 1× lane refines at a quarter budget while another zoom is the
  destination (it is not displayed then), and its mirror is not redrawn. A
  half-built back slot is abandoned as soon as the front is adequate again
  (zoom reversed, scrolled back), so idle maps stop requesting frames.

## 2. Shadow field: world-anchored and stable

**Problem.** `ShadowSamplingGrid::configure` fit the span (texel density)
exactly to the visible receivers' bounding box. Nearly every scroll step
changed that box, so every page was redrawn and every retained static pixel
invalidated. Camera transactions (1×) and visual frames (zoom) fit different
boxes, so they invalidated each other on every call at any zoom ≠ 1×.

**Industry practice.** Shadow maps for top-down/RTS cameras use a fixed
world-space texel density with texel-snapped, world-anchored pages/cascades
(stable cascaded shadow maps, virtual shadow maps). Moving the camera only
adds/drops pages; it never changes how existing texels sample.

**Implemented.** Receivers come from one *region of interest* (`update_roi`:
the 1× region around a camera quantized to 128 px, at the displayed/destination
zoom's ladder step), shared by both lanes: hidden 1× transaction draws use the
displayed lane's zoom, so the two lanes can no longer alternate the field. `configure_stable` keeps the span inside a
hysteresis band (refit only when the needed extent leaves `[want, 1.45×want]`,
choosing 1.15×want rounded to an even span). Pages then slide incrementally through the existing
page-content machinery. A `sampling_identity` (span, light basis, wrap, scene)
replaces the `shadow.builds` counter in raster and mirror validity.
`C3X_RENDERER_SHADOW_TIGHT_FIT=1` restores per-view fitting.

Trade-off: shadow texels are up to ~1.5–2× coarser than the old exact fit at
1×, more at high zoom (span tracks the destination zoom ladder). 9-tap PCF
hides most of it; if it is visible at 3×, raise page count or quality texels.

## 3. City lights and body placements: stable selection

**Problem.** Lights were selected from screen-visible cities; any change
invalidated all rasters. Body placements were recomputed per region view.

**Practice.** Select light sources for the area that can receive them
(receiver region + light reach), and only re-upload when the set changes
(clustered/tiled lighting does the same per tile).

**Implemented.** Lights and body placements come from the ROI (which includes
a 128 px light-reach margin, ≥ the 0.85-tile city light radius); selection
changes no longer invalidate pixels. The CPU light field is no longer rebuilt
every frame when the selection, night factor and GPU-field owner are
unchanged (`Gpu::light_uploads` serial guards other uploaders).

## 4. Frame pacing: vsync-locked, not timer-driven

**Problem.** A 16.667 ms waitable timer drove ambient frames and
`Present(0,0)` handed them to DWM immediately. Timer jitter and the beat
against the real refresh rate (59.94/60.00 Hz) caused periodic doubled and
dropped frames — judder at a nominal 60 — and frames started at arbitrary
vsync phase.

**Practice.** Flip-model swap chain with
`DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT`; the render loop waits on
the frame-latency object *before* starting a frame and presents with sync
interval 1 (Microsoft "Reduce latency with DXGI 1.3 swap chains"; standard
in shipping engines).

**Implemented.** `PresentationPermit::wait` blocks on a duplicated latency
handle outside the renderer gate; the helper cadence uses it as its pacer
(`c3x_renderer_trial_visual_wait`) and keeps the timer only as a fallback.
Grants are counted, so the worker and cadence thread acquiring concurrently
cannot leak a latency count.
Present uses sync interval 1 with maximum frame latency 2 (CPU preparation of
frame N+1 overlaps GPU work of frame N). `C3X_RENDERER_LEGACY_CADENCE=1`
restores timer pacing, `Present(0)` and latency 1.

## 5. Per-frame diagnostics

**Problem.** At the default trace level (1) every visual frame formatted
~6–10 large records (`sprintf` of 1–2 KB) and sent "important" records to
`OutputDebugStringA` (expensive, and much worse with a debugger attached).

**Implemented.** `fresh-callback`, the scene-phase/work blocks and
`frame-preparation-ready` require `C3X_RENDERER_TRACE=2`. Aggregate and
failure records are unchanged.

## Switches (all default to the fast path)

| Variable | Effect |
| --- | --- |
| `C3X_RENDERER_STATIC_LEGACY=1` | Synchronous full-quality static redraws (old behavior, incl. light-set invalidation and zoom mirror invalidation) |
| `C3X_RENDERER_BOOTSTRAP_SCALE` | Low-resolution first image after jumps (default `0.5`; `1` full resolution; `0` disables: jumps refine synchronously) |
| `C3X_RENDERER_REFINE_PIXELS` | Fixed refinement budget per frame instead of the adaptive one |
| `C3X_RENDERER_SHADOW_TIGHT_FIT=1` | Old per-view shadow fitting |
| `C3X_RENDERER_LEGACY_CADENCE=1` | Old timer pacing, `Present(0)`, latency 1 |
| `C3X_RENDERER_TRACE=2` | Restores per-frame trace records |

## What to test

Use the normal staged build (`BUILD_RENDERER64.bat`) and the short diagnostic
capture. With water, waves and reflections on:

1. **Idle** at 1× and 2×: steady 60, no periodic hitch. Compare against
   `C3X_RENDERER_LEGACY_CADENCE=1`.
2. **Zoom** one notch and a fast multi-notch spin in/out: motion stays smooth;
   the image is slightly soft while moving and snaps sharp shortly after
   landing. Zoom-out edges should never be black.
3. **Scroll** at 1×, 1.25×, 1.75× and 3× (edge and arrow keys), including long
   continuous scrolls past the guard band and across the world wrap seam:
   no periodic hitch; no seams between old pixels and new strips.
4. **Jumps** (minimap click, center on distant unit): the destination appears
   promptly (briefly softer), then sharpens within a few frames.
5. **Night with several cities**, crossing cities in/out of view: lights stay
   correct and no full redraw hitches.
6. **Turn change** (hour/season change): the lighting switch may land a few
   frames after the dynamic layers; no freeze.
7. **Reveal while moving units**: newly revealed terrain may appear a few
   frames late (progressive), never as black holes.
8. Config-off and `C3X_RENDERER_STATIC_LEGACY=1` still behave as before.

Watch for: soft edges at a jump that never sharpen (refinement stuck — check
that frames keep presenting), shadow seams, and units/water misaligned by a
pixel after fractional-zoom scrolling.

Host contract tests were updated to the new behavior and pass on macOS:
`test_scroll_region.py` (odd/fractional reuse with snap),
`test_static_raster_state.py` (lanes, promote, stale preview, discard),
`test_static_dependency_reuse.py` (stale-on-reorder; stable shadow span under
scrolling, refit on real extent change), `test_native_visual_cadence.py`
(grant counting, `Present(1)`), and `test_fresh_shared_submission.py`
(region-of-interest reuse/rebuild). Windows-only witness scripts that read
`static_rasters` slot metrics still assume the old two-slot layout.

## Remaining opportunities (ranked, not implemented here)

1. **Shader cost (the ~130 ms floor itself).** Runtime shaders come from the
   pinned pack (`packs/Renderer64CutoverControl` or
   `C3X_RENDERER_SHADER_SOURCE_ROOT`), not the repository copies, so they were
   not edited. Highest-value changes: 9-tap PCF → 4-tap hardware
   `SampleCmp` bilinear PCF; terrain material layers collapsed into packed
   atlases (20–35 samples → ~10); hydrology's 4-tap height-to-normal → a
   normal map; skip triplanar off steep slopes; depth-only prepass with
   `EQUAL` testing for alpha-tested vegetation/decals to stop shading hidden
   fragments. With progressive refinement this now buys *time to full
   quality*, not frame rate.
2. **Territory borders** redraw every bordered tile's full terrain mesh every
   frame (one draw + constant update each, hundreds of draws in a developed
   empire). Retain a border overlay keyed by camera/zoom/border revision and
   composite it under units.
3. **Water** is fully shaded every frame. Cache the camera-independent part
   (shore distance, depth, bed color, static reflection) in the retained
   layer and evaluate only the animated normal/specular per frame, or shade
   water at half resolution with a depth-aware upsample.
4. **Retained composition** still copies the full display into its retained
   buffer every changed frame and walks the whole native graph; make the copy
   demand-driven and cache map-independent HUD as an alpha layer.
5. **Game thread** still runs Civ III's full visible-map traversal for each
   native map draw (≈4 ms + ≈2.5 ms capture). Skip native rasterization when
   custom rendering owns the map and keep only the anchor/fact enumeration.
