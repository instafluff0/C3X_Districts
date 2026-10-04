# Renderer64 performance overhaul — October 3, 2026

Goal: ~60 FPS during idle, scrolling, zoom and map jumps on the Windows VM,
treating memory as plentiful. This note records each major problem found in
the inherited build, the industry-standard remedy, what was implemented, the
switches that restore the old behavior, and what remains. It supersedes the
"no FPS tuning before the integration gate" guidance for the items below.

The original overhaul was not run by its author (no Windows/D3D11 host). All three build
products (x64 renderer DLL, x64 helper, x86 bridge) were syntax/semantics
checked with mingw-w64 against a baseline of the unmodified tree: the only
diagnostics are the same pre-existing GCC-vs-MSVC differences. Strict
`-Wall -Wextra -Wshadow -Wconversion` reports nothing on changed lines.

The completion pass below was built and checked on Windows/D3D11. Its checks
establish correctness for the tested contracts, not a measured gameplay FPS gain.

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
background work can be reduced when frames miss the target. Pixel/page budgets
are estimates; an expensive individual draw can still exceed a frame interval.

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
  projection) is affinely resampled into the frame — bilinear color and locally
  reconstructed depth — and the dynamic layers (water, waves, units, borders) render at the
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

## Second round (same day)

**Runtime shader root.** The current build needs resident vertex entries
that the pinned `Renderer64CutoverControl` pack lacks. Launching without
`C3X_RENDERER_SHADER_SOURCE_ROOT` failed shader setup every frame (black map,
overlay smearing). The runtime and `BUILD_RENDERER64.bat` now default to the
local prepared pack `Renderer/packs/Renderer64ResidentRuntime` when present
(generate it with `prepare_resident_submission_shaders.py --baseline
Renderer/packs/Renderer64CutoverControl --out
Renderer/packs/Renderer64ResidentRuntime`). `BUILD_RENDERER64.bat` is now CRLF:
`cmd.exe` label lookup fails in LF-only batch files once offsets shift.

**Dirty-rectangle repair (gameplay).** A local world change (fog reveal,
road, irrigation, city growth, ownership) used to mark the whole retained
raster stale and re-render it. The displayed raster now diffs its retained
contributor proofs against the current contributors (added, removed, changed
content or tile visibility), clears and redraws only those records' screen
rectangles plus a 96 px shadow/light reach, then rebuilds its proofs. Broad
changes (over 45% of the raster) still refine progressively. Industry analog:
dirty-region invalidation in retained-mode UI and tile caches.

**Water mirror reuse while scrolling.** The mirror key included the camera,
so every scroll step re-rendered the reflected scene. The water pass samples
the mirror at `screen + NativeReflectionTarget.zw`, so a camera-only change is
an offset: while the camera moves (and within 160×96 display px) the mirror is
reused with that offset; the exact mirror (with unit reflections) is redrawn
once the camera is still for 150 ms.

**Territory borders retained.** Every bordered tile's full terrain mesh was
redrawn every frame. Borders not near water (conservative 2-tile test) are now
drawn into the retained static raster with its strips (scissored, region
margins); only near-water borders stay in the per-frame pass above animated
water. Ownership is part of each contributor key, so changes repair the raster.

**Shadow region keyed to the zoom destination.** Tracking the animating zoom
refit the shadow field (a full page redraw plus refinement restart) at every
ladder step crossed during one wheel notch. It now refits at most once per
destination; during zoom-in the screen edges can lack dynamic-layer shadows for
the ~166 ms animation.

**Smaller jump bootstrap.** The low-resolution jump image covers the view plus
a 128 px band instead of the whole guarded region (~25% less first-frame work).

**Per-frame CPU trims.** Diagnostic `C3X_SANDBOX_*` switches were read with
`GetEnvironmentVariableA` (process-wide lock and scan) per layer and per
record; they are memoized (`render_core/process_environment.h`). The full-screen
copy of every presented frame into a readback buffer now happens only for
input replay (`C3X_RENDERER_RETAIN_DISPLAY`, set by `replay_inputs`) or after a
readback request.

## Completion pass: shader work, shadow preparation, water and HUD

This pass changes `sandbox/fresh_pipeline.h`, `sandbox/terrain_material_fast.h`
and `native/gpu_spatial_composition.h` as one connected implementation.

- **Terrain texture work:** skip material families whose blend contribution is
  exactly zero, and skip the six cliff texture samples when cliff exposure is
  zero. Preserve all contributing grass/plains/tundra/desert/hill textures and
  the authored color equations. Explicit texture gradients preserve mip selection
  through the material branches. This reduces unnecessary fetches without a new
  asset format, offline atlas conversion or different terrain art.
- **Shadow filtering:** use four hardware bilinear comparisons on shallow
  receiver gradients, retaining the nine individually corrected comparisons on
  steep receivers and at page boundaries. Check R32_FLOAT comparison support;
  unsupported devices retain the original filter. The small filter-kernel change
  needs visual comparison. See Microsoft's [comparison sampling contract](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-samplecmp).
- **Distant-view shadow preparation:** draw at most two missing pages per scene
  invocation, with center pages first. Already completed pages survive subsequent
  invocations. Missing pages are explicitly unpublished and sample as unshadowed,
  so recycled slices cannot cast shadows from an unrelated location. Keep visual
  frames running while pages are pending; start full-quality scenery refinement
  after shadow completion. Completion invalidates the temporary preview once.
  Metadata/proof refusal retains the existing synchronous fallback.
- **Refinement:** remove the emergency unbounded full redraw after eight
  refinement restarts. Repeated camera input keeps its preview and bounded work.
- **Water:** at a settled native-resolution view, retain water illumination and
  shadow visibility in an HDR texture. Reuse it while camera/projection, visible
  content, environment and shadow contents remain unchanged. Normals, Fresnel,
  marine resources, reflections, specular and foam remain animated. During motion
  use the direct shader, avoiding a new cache preparation pass on every camera
  step. Unsupported sample/scale modes also use the direct path. The cache costs
  eight bytes per guarded viewport pixel and introduces normal half-float storage
  precision. It does not cache the complete animated water image.
- **HUD:** retain exact resolved native-word/full-color pixels for a complete
  pointwise HUD program. Conservative dependency tracking admits a pixel only
  after both outputs cease to depend on the changing map before-images. Remaining
  pixels execute the same ordered native operations. Classify dependence once
  per program, including pixels that cannot be cached. Command/placement/source
  recompilation clears the cache; programs containing interpreter boundaries
  retain their existing execution. This is an exact packed-color cache, avoiding
  the rounding differences of substituting ordinary alpha blending. Optional
  allocation failure keeps the old path. Storage is twelve bytes per viewport
  pixel, charged to existing budgets. Three R32_UINT slices use the baseline
  [D3D11 typed UAV load formats](https://learn.microsoft.com/en-us/windows/win32/direct3d11/typed-unordered-access-view-loads).

Shader adaptations run against the selected pack during runtime shader creation.
No ignored art/shader pack, native patch table, gameplay state or reference image
was edited. Existing C3X capture and presentation boundaries remain;
`required_user_action: []`.

### Verification and limits

- Windows MSVC bridge, renderer and helper build; isolated startup check.
- Nineteen affected runtime pixel-shader variants compile against the prepared
  local pack on the VM, including water lighting and terrain/reflection variants.
- GPU HUD comparisons exactly match the ordered compositor for RGB555 and RGB565
  over six changing underlays. Each format's 256×192 fixture caches 28,900 pixels;
  the other pixels remain dependent. Existing alias, source and target-rebinding
  cases also pass. This is a correctness witness, not a HUD speed measurement.
- Both shadow filters pass retained-versus-rebuilt GPU comparisons across camera,
  caster, light, wrap, asset and device changes. Unpublished pages return fully lit
  visibility rather than stale atlas values. These are comparisons within each
  filter, not a claim that the two filters produce identical edges.
- Host tests cover partial completion, caster replacement, invalidation, wrapping,
  raster promotion and retained dependencies.

Evidence is bounded to `Renderer/.cache/performance-completion/`. No game was
launched or installed during this pass. Fullscreen late-game idle, scroll, zoom
and jump latency, refinement settling time, water-cache visual parity and shadow
edge quality still need a current-candidate gameplay/scene comparison. No new FPS
number or 60 FPS guarantee is claimed.

### Water stripes during zoom: depth reconstruction

A live screenshot showed regular horizontal stripes through water during zoom.
The preview resampler used the nearest old depth pixel while the water pass
rendered at the current projection. Even a flat map plane has a screen-space
depth slope: nearest sampling turns it into steps, making the water fail its
depth test on alternating rows.

`render_core/linear_target.h` now reconstructs depth at the fractional source
coordinate using the smaller agreeing neighbor slope on each axis. This preserves
planes, limits interpolation at depth discontinuities and keeps clear depth
exactly 1. The settled whole-pixel restore path and animated water shader remain
unchanged. Only preview composition adds four nearby depth reads; there is no
extra geometry pass, target allocation or CPU/GPU synchronization.

The standalone Windows GPU regression `test_zoom_water_depth` runs 72 cases:
six intermediate zooms, full/half-resolution sources, positive/negative depth
adjustments, both source slots, opaque silhouettes and empty areas, using the
current water/underlay separation at a 128-pixel native tile width. The old
nearest-depth control wrongly rejects 269,532 water pixels; reconstruction rejects
zero eligible pixels, preserves 146,688 occluder checks and 148,224 clear-depth
checks. This reproduces and fixes the depth failure in a controlled scene;
confirmation against the reported live-game view is still pending. No new FPS
measurement is implied.

## Movement and reveal consistency repair

A supplied movement recording shows terrain/city corruption from about 2.53 to
2.87 seconds, followed by recovery and newly revealed scenery. Inspection found
two concrete defects in the retained raster path:

- Background refinement checked camera/environment identity but not the content
  of bands drawn on preceding frames. A reveal, removal or appearance change
  could therefore promote mixed scenery. It now validates the covered bands
  before continuing; unaffected bands retain their work.
- Local repair cleared the displayed raster before drawing all replacement
  layers. A later mesh/instance draw failure left that damaged raster available
  as the fallback preview. Repairs now use the lane's existing back target and
  copy only completed dirty rectangles, including depth, after every draw
  succeeds. Failed repairs preserve the displayed color and depth.

`test_static_refinement_consistency` executes the production refinement and
repair code with controlled content changes and late draw failures: ten cases
across both zoom lanes and twelve repair cases across all four slots. Both
regressions fail against the preceding implementation and pass with these fixes.
The repair adds a GPU copy of the dirty rectangles only; it creates no additional
full-screen target owner. The fresh live reveal check below passes; the exact
city-bearing scene in the supplied recording has not been replayed.

A subsequent disposable-game capture (`20261003-170210`) completed two inland
moves and three zoom/text checks without logged renderer failures, but sampled
frames still caught a brief return to the first unit's source position during
the reveal. This isolated a third defect in retained composition: rebuilding a
native view after its map sampler retired lost the map's projected classification
and could return to the older canonical publication. The completed projected
image now stays eligible through that handoff, including supported keyed native
and unit overlays. Native UI changes still apply; retired callbacks stay retired.

The earlier capture is diagnostic evidence only: the user reported possible
concurrent play during testing. A fresh run after they closed Civ III supplies
the final live evidence below.

The extended `test_ready_frame_composition` GPU regression reproduces the rewind
before the fix, both directly and through an overlay. It checks the repaired
paths with ordinary and map sources, unchanged callback counts, current overlay
pixels and preserved final-pose pixels. The retained compositor's 657 exact GPU
oracles and the 80-camera lifetime checks also pass. The asynchronous native
fixture passes 32 camera transitions and 24 pose changes with zero compositor
errors. Its native hook stubs were updated for current presentation fields and
an observer name collision; its bounded first-map allowance is now 120 seconds
(the successful cold preparation took 43 seconds).

Fresh disposable-game capture `20261003-173427` completed both inland moves,
three zoom/text checks and all 13 posted commands. Review of 66 arrival/reveal
samples found no return to the old unit position or damaged revealed tiles.
The first unit remains at its destination through the replacement map handoff.
The complete log has no renderer failure markers. Save/configuration cleanup
and absence of remaining game/helper processes were verified. The matching
bridge, renderer and helper are staged with candidate and game-link hashes
checked. Local receipts and images are under
`Renderer/.cache/movement-glitch-review/`. These sampled checks establish no FPS
result or blanket gameplay acceptance.

Validation limitation: two pre-existing `test_frame_publication` host fixtures
still fail compilation because their worker stubs lack current completion and
residency members. The focused regressions, transition category suite, GPU
oracles and asynchronous/live checks above passed; no injected source changed.

## Switches (all default to the fast path)

| Variable | Effect |
| --- | --- |
| `C3X_RENDERER_STATIC_LEGACY=1` | Synchronous full-quality static redraws (old behavior, incl. light-set invalidation and zoom mirror invalidation) |
| `C3X_RENDERER_BOOTSTRAP_SCALE` | Low-resolution first image after jumps (default `0.5`; `1` full resolution; `0` disables: jumps refine synchronously) |
| `C3X_RENDERER_REFINE_PIXELS` | Fixed refinement budget per frame instead of the adaptive one |
| `C3X_RENDERER_SHADOW_TIGHT_FIT=1` | Old per-view shadow fitting |
| `C3X_RENDERER_LEGACY_CADENCE=1` | Old timer pacing, `Present(0)`, latency 1 |
| `C3X_RENDERER_TRACE=2` | Restores per-frame trace records |
| `C3X_RENDERER_TERRAIN_MATERIAL_REFERENCE=1` | Sample all original terrain material families |
| `C3X_RENDERER_SHADOW_PCF_REFERENCE=1` | Original nine-comparison shadow filter |
| `C3X_RENDERER_SHADOW_PAGES_PER_FRAME` | Missing-page budget per scene invocation, default 2, clamped to 1–25 |
| `C3X_RENDERER_WATER_LIGHTING_REFERENCE=1` | Recompute water lighting per frame |
| `C3X_RENDERER_HUD_CACHE_REFERENCE=1` | Execute all pointwise HUD pixels each frame |

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

## Remaining opportunities after the completion pass

1. **Further shader work:** pack contributing height/specular channels, precompute
   hydrology normals, and investigate depth prepasses where overdraw dominates.
   The completion pass addresses unused terrain families, flat-ground cliff work
   and shadow filtering; it does not repack assets or add a universal prepass.
   Shader cost still affects motion, dynamic passes and time to full quality.
2. **Near-water territory borders** still redraw per frame (see above).
3. **Water during motion:** lighting retention currently benefits settled views.
   Wider world-space retention or depth-aware lower-resolution shading could
   reduce moving-view cost if measurements still identify water as dominant.
4. **Retained composition CPU traversal:** the new HUD cache avoids repeated GPU
   pixel programs. It does not remove all CPU graph/dependency visits.
5. **Shadow preparation:** drawing is incremental between pages. Caster selection,
   instance preparation and one unusually expensive page can still cause a stall.
6. **Game thread** still runs Civ III's full visible-map traversal for each
   native map draw (≈4 ms + ≈2.5 ms capture). Skip native rasterization when
   custom rendering owns the map and keep only the anchor/fact enumeration.
