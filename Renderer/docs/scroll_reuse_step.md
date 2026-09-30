# Bounded scrolling reuse candidate

Status: source candidate for Astra review; no installation or reference replacement. The native camera pilot proves projection thrash, not a 1.25x performance win. Existing scene/geometry epoch invalidation remains intact.

The candidate uses exact integral projected shifts (`zoom * native camera delta`) inside the retained guard. Fractional phases redraw without rounding native anchors. Strip fills and restore offsets use that projected shift. Retained depth uses the actual captured depth translation/origin rather than assuming native camera Y is depth; clear depth remains exactly 1. Environment and immutable selected-light changes invalidate retained pixels. Reflections still redraw per camera.

Validation: both production-recipe DLL arms and the common client compile with O2/W4/WX. Five focused host checks pass. The real-D3D restore oracle passes 18 cases at 1/2/4 samples, exact HDR color/alpha and clear depth, occupied depth within one D24 code, foreground ordering, and three known failing legacy controls. The existing real-D3D depth-basis test passes. No injected sources changed and no new Civ III patch symbols are needed (`required_user_action: none`).

## Native pilot result

Read-only accepted C7 inputs: all 193 assigned runtime source hashes match the primary checkout and starting commit. The control DLL restores those sources, adding only untimed cache-reset/metric wrappers. Evidence stays in the isolated worktree under `Renderer/.cache/scroll-reuse-preflight/` and `Renderer/.cache/scroll-reuse-pilot/`.

The six-view fixture uses copied native-role captures, begin/readiness/adoption, absolute anchor origins, 128/64 native tiles, source-derived slow edge increments 4/2, positive/negative motion, a 128/64 strip destination and return. The guest/client/scene are 2240x1260 at 60 Hz. All visual effects remain enabled. Actual pass counters show zero unit, reflected-unit and unit-shadow submissions: this proves the cities/coasts preparation blocker, not busy-actor capacity or injected-game behavior.

At 1.25x every preparation and display stage executes one static full draw and zero scrolling restores. Geometry epoch stays 2 for the small motion inputs and advances to 3/4 on capture membership changes. Projection invalidation and membership invalidation are separate causes.

| Quiet changing-view measurement (five samples) | Candidate | Accepted control |
| --- | ---: | ---: |
| First correct view, mean | 264.82 ms | 277.90 ms |
| First correct view, median | 248.40 ms | 282.59 ms |
| Preparation, mean | 117.71 ms | 130.39 ms |
| Display draw submission, mean | 142.27 ms | 140.76 ms |
| Views exceeding 16.667 ms | 5/5 | 5/5 |

These small sequential pilots establish no FPS, tail, or statistically reliable speedup claim. Cold destinations are separate. Diagnostic/quality runs alter pacing and include instrumentation or untimed captures; the 1x unbuffered pass-count run is especially unsuitable for timing qualification.

At 1x the candidate reuses the static region for three small pans and executes two strip fills per pan. Independent same-time full redraws establish a substantial existing baseline error and its reduction:

| Positive 4/2 pan comparison | Candidate | Accepted control |
| --- | ---: | ---: |
| Changed D24 pixels | 187,984 | 2,643,796 |
| D24 pixels beyond one code | 31 | Broad legacy camera-Y shift |
| Changed BGRA pixels | 36,762 | 1,339,836 |
| Mean absolute BGRA difference | 0.01427 | 9.21710 |

Most candidate depth changes are +/-1 code. The other small pans have 29/42 samples beyond one code. Larger outliers lie on depth/coverage discontinuities and can exchange neighboring surfaces; they are not dismissed as exact or proven random. Existing crops show the candidate retaining the forest/coast with isolated leaf-edge differences, while the control loses large farm/route regions through incorrect occlusion. Alpha and stencil match in all pairs. The candidate still has color differences, including 1,320 pixels exceeding ten channel levels in the 4/2 pan; exact image parity is not claimed. Non-1x reuse remains unreachable in this route. Odd integral projected shifts can change shader derivative-quad phase, so their visible result still needs a bounded comparison when separate raster states make that path reachable.

Full/full 1.25x pairs have only 68-867 changed color pixels and one changed depth pixel in three of six views. Examples (depth coordinates include the four-pixel guard): `(1088,599): 6506035 -> 6473897`, `(333,1137): 6020390 -> 6128987`, `(2153,15): 7027507 -> 6995251`. The first and last differ little or not at all in output color; the middle switches one forest/decal edge pixel. Both paths fully rasterize, so these exceptions cannot establish an error in scrolling sample reuse. Fragment ownership is inferred from nearby owner records, not measured.

`bounded-review.json` preserves per-pixel values and dominant deltas; each run has source/binary/scene/shader hashes, copied sources, logs, child PID and matching completion receipts. The first draft `candidate125-pilot` is explicitly discarded for fixture-coordinate/timing/receipt errors. `vm-release.json` confirms all six owned launcher PIDs and renderer/compiler/game processes are absent. VM ownership was released to Astra; no further VM work is authorized under that reservation.

## Production call sequence and canonical consumers

1. `RendererState::render` in `native/c3x_renderer.cpp` calls `c3x_renderer64_render_fresh(frame,target)` at its default 1x during preparation.
2. `capture_gpu` copies the completed canonical map into `PublishedMapFrame`; camera adoption publishes it through `gpu_composition_session.h::publish_source` into the native immutable map/retained source.
3. `retain_visual_map` installs a projected sampler that calls the same fresh pipeline at the requested display zoom. One shared `static_region` is replaced on each 1x/non-1x alternation.
4. `retained_composition.h::evaluate` preserves canonical native save/restore images. `evaluate_projected` uses the owned completed canonical publication if a camera retires before its first display and no projected output exists.

A zoomed image cannot replace the canonical publication, and a pending camera cannot be declared ready before its actual publication is available.

## Smallest next runtime patch

Introduce exactly two small retained-raster states inside the existing pipeline: canonical 1x and one current display projection. Retain one world/content lease, geometry selection, shaders, assets, shadow field and drawing code. Select canonical for 1x; a changed non-1x zoom replaces only the display state's identity and reuses its allocations. Do not duplicate `SandboxFreshPipeline` or accumulate an arbitrary zoom LRU.

Each state owns `static_region` color/depth and its projection/layout, static validity, scene revision/geometry epoch, shadow identity, environment/light proof, region camera/anchor translation, covered rectangle and actual depth translation/origin. Retaining `reflection_static` color/depth plus reflection camera/depth/receiver proof in each state also preserves the static reflection; a static-only first patch may keep one reflection target and conservatively redraw it on projection changes.

Keep `static_cache` as shared viewport restore scratch, and always restore it when its writer state changes, even at an unchanged camera. Keep live `reflection` shared and copy the selected `reflection_static` into it on a state switch even without units or a rebuild. Shared reflected-terrain material targets need one writer key including projection, generation, camera/anchor transform, depth basis/origin, reflection extent/scale and capture rectangle. Two independent valid flags cannot describe one overwritten target. The unused main material/terrain experiments have no active prepare callers and need no extra allocations.

Glow, bloom, aquatic targets/bounds, streams, unit poses, borders, output transfer and final BGRA work target remain shared scratch consumed in the serialized draw. Native-anchor selection conservatively covers both zooms; the shared world shadow field retains its existing coverage rules.

| Added texture storage at 2240x1260 | 1 sample | 2 samples | 4 samples |
| --- | ---: | ---: | ---: |
| Second 2888x1652 static region | 54.599 MiB | 109.199 MiB | 218.398 MiB |
| Second 846x478 static reflection | 4.628 MiB | 4.628 MiB | 4.628 MiB |
| Recommended two-target total | 59.227 MiB | 113.827 MiB | 223.026 MiB |

The default one-sample two-target increment is 62,104,368 bytes. Views/aliases add no backing textures. A second viewport `static_cache` would unnecessarily add another 32.621/86.989/152.231 MiB. Full-resolution reflections increase the added reflection target to 32.944 MiB. The 2/4-sample numbers are allocation arithmetic; current reflected-terrain target allocation is limited to sample1 and needs verification before claiming complete scene support.

Invalidate both states on scene/geometry generation replacement, environment/light change, resize/sample/layout change, device/config reset and independent-raster diagnostics. A per-state environment/light proof on activation is an alternative to eager dual invalidation; the shared `previous_hour` check alone is insufficient. Depth-origin, phase, guard and anchor failures retire the affected state. Preserve safe-redraw behavior and the scene epoch policy.

Use references to selected state; never copy owning targets by value. Allocate the display state lazily; release both once at reset and clear scratch writer identities. Preserve serialized immediate-context ownership and reflection-binding RAII. Existing retired camera/viewer/publication guards must return frozen before touching either state. Old native publications remain owned completed BGRA images independent of these HDR caches.

Canonical preparation may restore its valid region, fill required strips and render current-time water/resources/units, bloom, transfer, city overlay and visibility without a full static raster solely because display previously used 1.25x. Genuine invalidation still redraws. This directly removes the measured destructive projection alternation while preserving native save/restore and retired-before-first-display behavior. This structural patch has not been implemented in this step.
