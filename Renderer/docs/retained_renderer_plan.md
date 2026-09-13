# Retained renderer program

Current status and preserved evidence for the retained renderer. Planned work
is not evidence that implementation or performance targets pass.

## Current status — local finishing damage retained; compiled-content validity next

Local region validity passes the dense edit comparison: the four-edit sequence
is 16.8–16.9% faster, with exact independent full redraws. Distant edits average
425–465 ms control versus 103–116 ms candidate. Visible edits remain slow at
1.85–1.90 seconds, dominated by geometry preparation. A separate navigation check
is unchanged within one matched pair (193.4 / 193.8 ms); the below-100 ms navigation
gate and native presentation remain unmet.

The previously accepted shadow-only optimization remains retained: the user
explicitly accepted its repeatable 13.3–13.6 ms navigation saving despite the
previous 10%/20 ms selection rule. These are different workloads; do not add the
percentages. Both implementations remain scoped opt-in paths, with no staging,
installation or live-game claim. Tooling is complete and failed raster/array
experiments remain closed. The 100x100 fixture uses realistic synthetic density,
not captured live state.

[The architecture](renderer_architecture.md) defines the destination;
[the execution contract](autonomous_renderer_execution.md) governs continuation;
[the benchmark workflow](benchmark_workflow.md) specifies validation. This status
is the single active task record.

### How this reaches fast rendering

We are completing reusable world/instance inputs and have begun explicit pass
ownership; we have not yet made general view submission efficient. The dense
standalone workload now averages 147–154 ms in the latest paired runs, against a first below-100 ms gate
and then 33 ms. These are workload gates, not promised native frame rates.
The throughput path is resident content → selected compatible draw lists → fewer
repeated render passes and output transfers → exact caller-driven native
consumption. Local revisions keep that path incremental when game state changes.
Raster reuse can skip work, but newly exposed views must also be inexpensive.

The current resource representation is a prerequisite, not the decisive speedup.
Order-preserving instance submissions were implemented and tested, then removed
after mixed whole-workload results. The common-depth task addressed the depth/overlap contract needed for a coherent
dynamic scene pass to replace repeated region rendering. Both pre-raster
rebasing and post-raster import failed exact legacy ordering; the GPU witness
below establishes that stored depth alone cannot recover the new rounding.
The user authorized small reviewed depth-rounding differences on 2026-09-13.
The common producer contract is retained. Both coherent consumers were exact but
slower and removed; the causal breakdown led to the retained local finishing change below.
Failed numerical post-raster rebasing remains closed.
Latest animation completion averages 78–83 ms and much of that is GPU/readback wait;
we have not isolated all of that wait as transfer cost. A few milliseconds saved
in pose maps/bindings cannot by itself meet the navigation target. Do not chain
further helper optimizations if batching leaves that dominant cost intact.
The previously rejected texture-array region batch remains closed; it did not
remove the underlying repeated region work enough to help.

Native publication identity is connected, but live behavior remains unverified.
The renderer supplies ready compatible work when Civ III calls; it never notifies
the game. Exact-camera overlays/picking constrain camera handoff now, so asynchronous
bookkeeping cannot be presented as a substitute for faster view rendering. The
strategic live checkpoint still requires staging/install/game authorization.

### Destination checkpoints

The user's clarified objective keeps all six capabilities in scope. This table
records architectural gaps, not a second task queue or completion percentage.

| Destination | Current foothold and remaining gap |
| --- | --- |
| Complete persistent scene/world-instance database | A bounded canonical captured-appearance owner now feeds compiler lookups and retains local revisions; resident compiled-content bindings are now consumed; complete appearance capture and sparse object/action ownership remain unfinished. |
| General spatially selected draw lists | Static draw/receiver/caster/region consumers now read compact resident occurrence records; general spatial selection feeding compatible batches remains unfinished. |
| Systematic local revisions and invalidation | Local topology/dependency observations, retained-region validity and finishing damage now preserve unaffected work; compiled-input owners still have broad revision invalidation. |
| Broad batching/instancing with explicit passes | Animated resources now draw shared resident source meshes with separate pose/placement inputs through body/shadow passes; compatible instance submissions and general batching remain unfinished. |
| Completed asynchronous Civ III presentation | The injected compositor now supplies lifecycle/visibility identity and consumes exact queued or compatible ambient work through the existing owner; general asynchronous camera handoff and live presentation verification remain unfinished. |
| Camera movement with bounded reconstruction/rendering/readback | Retained overlap and region reuse exist; latest candidate dense navigation is 147–154 ms and remains above the first 100 ms gate. |

The publication handoff, captured-appearance owner, resident-content bindings and
compact occurrence records are retained. The first static terrain batch representation
was rejected after exact but unhelpful whole-workload results. Native request identity
is now connected and verified in source/replay. Shared animated resource geometry
is structurally retained; common scene depth and local finishing damage are retained.
Local validity of reusable compiled content is selected next. Other gaps remain explicit dependencies,
not tasks to pursue concurrently.

### Prioritization correction

The user explicitly requested established RTS rendering approaches to guide
priorities rather than a chain of nearby experiments. Apply the destination's
persistent world content/local revisions, spatial selection with batching and
instancing, explicit static/dynamic passes and bounded GPU ownership as the
basis for implementation selection. Measurements choose among these mechanisms
and validate complete changes; a large phase does not justify successively
trimming every nearby operation. Preserve useful existing content reuse rather
than adding more completed-image cache policy. Keep readback/publication at the
Civ III bridge, with its exact-camera/overlay constraints.

The current upstream [0 A.D. source review](0ad_renderer_review.md) supports
shared immutable meshes, locally dirty terrain patches, material variants and
preparation shared across passes. Those are concrete mechanisms for the existing
architectural destination. It also distinguishes world-content sharing from our
unsuccessful screen-region batching. No external engine code was imported and no
new performance claim follows from that review.

### Architectural selection after the user's course correction

The source review against all six capabilities finds persistent compact topology
and shared tile meshes, but no complete world appearance/instance owner; view
assembly still rebuilds captured occurrence/layer lists. The contributor index
serves cache/shadow queries rather than a general selected draw-list owner.
`submit_geometry` combines selection, recursive passes and GPU bindings, limiting
compatible batching. Local validity improvements are useful but do not complete
these responsibilities or make newly exposed content cheap to draw.

The first architectural boundary selected after that review was **caller-driven
publication selection**. Civ III owns render demand; the renderer prepares work and supplies a
compatible ready result at the next native render call. The user's clarification
explicitly rules out worker-completion notification/redraw requests. The proposed
notification code was removed before any build or native changes. Existing game
animation scheduling is preserved, not converted into a completion channel.

### Local finishing damage — retained, modest measured improvement

Conservative posed body/shadow bounds now carry four native pixels of filter
support into the existing reconstruction, display conversion and copy owners.
The complete guarded MSAA source and hardware resolve remain unchanged. Each
output starts from the immutable static bitmap, so removal and old poses cannot
leave stale pixels. The superseded unconditional whole-cell finishing path is
replaced inside the existing prepared-resource pass; no new cache, target, shader
or runtime switch was added. Region metadata remains charged under the existing
4 MiB preparation cap.

**Architectural acceptance:** local damage now controls finishing and publication
copies as well as draw selection. The production GPU witness passes nine exact
HDR/MSAA/filter cases, including guard-only inputs, workgroup edges and thin spans;
six independent boundary redraws pass, including motion and removal/reappearance.
All five saved dense images in each matched pair are exact. Three zoom revisits
pass and all saved 128/160/192 images equal the retained common-depth control.
Minimum sampled contiguous headroom across candidate dense runs is 1,698.7 MiB,
well above 512 MiB; transient driver residency remains unmeasured. The complete
source guard and material/depth behavior are preserved.

**Performance acceptance:** two serial candidate/control pairs, each with two
assets-loaded cases of 14 requests on the dense 100x100 synthetic world at
2240x1192, width 128, waves/reflections off:

| Pair | Control mean (ms) | Candidate mean (ms) | Saving |
| --- | ---: | ---: | ---: |
| 1 | 157.758 | 153.864 | 3.895 ms / 2.47% |
| 2 | 152.822 | 147.206 | 5.617 ms / 3.68% |

Individual case means vary: candidate 146.75–157.93 ms, control 152.03–162.57 ms.
The targeted animation completion mean improves in both pairs (83.69 → 81.01 ms;
83.51 → 78.15 ms). Its completion wait falls from about 62 ms to 57–59 ms and CPU
bitmap copy from about 0.94–0.98 ms to 0.60–0.69 ms. These are overlapping endpoint
components, not additive GPU timings. Copied animation pixels fall to roughly
200,000, but that proxy is not the acceptance result. Retain the small complete
change for repeated whole-workload benefit and its end-to-end damage responsibility;
it is not the decisive throughput improvement or a passed navigation gate.

**Remaining limitation:** full hardware resolve, static backdrop restoration and
per-cell rendering still run. This path excludes views with waves. The 100 ms then
33 ms targets, general spatial batching and native presentation remain unmet. Do
not continue dispatch/copy helper tuning from this result. The causal diagnosis is
closed at its available precision, including the invalid VM GPU clock below.
Focused current-code resource integration passes 255 tests (one existing skip),
plus animation, zoom-return, scroll and removal replay. Implementation identity:
`a8cde6ba6738e35956613f7f336f08c23ef8d41e40ef5d7164e112abe29917fa`; DLL:
`1729f5a06540db1380002690c2aa9a48356dcf10df6137289111664f834fda45`. No staging, installation, injected
compilation or live-game claim.
Evidence: `Renderer/native/build/local-finishing-damage-20260913/comparison.json`,
`contracts.log`, `boundary.log`, `zoom.log`, and the four matched navigation directories.

### Single next implementation task — local validity for compiled terrain content

The user clarified that this is an **edit/rebuild evaluation**, not a navigation
speedup. Close it with evidence, then return to the unchanged-map navigation
bottleneck. Do not extend this task into successive edit-cache improvements.

**Next task:** complete dependency propagation from the existing natural river-page
owner into reusable terrain/natural compiled content, then replace the whole-map
revision in those content keys with complete local validity. Reuse the existing
world observations, compiler dependency records and bounded content owners. Audit
all river-page consumers, including the terrain compiler and natural relief, before
removing a global guard; keep global map/device/asset/reset identity intact.

- **Capability afterward:** unchanged resident terrain/natural content survives an
  unrelated authoritative edit, including content reused when entering a new view
  or zoom. This extends systematic local invalidation from raster results to the
  persistent content that must render on a cache miss.
- **Repeated work removed:** rebuilding reusable samples and shared world meshes
  solely because another part of the map advanced the topology revision. Existing
  visible-edit evidence rebuilds 81 tiles and takes 1.85–1.90 seconds; that motivates
  examining compilation, but is not a promise that all 81 builds are unnecessary.
- **Concrete dependency:** `NaturalWorld` currently clears its 16 river pages on
  every topology revision. Those pages read a 16x16 support area and terrain-height
  lookups outside the tile compiler's recorded queries. Shared natural and ground
  content keys still include the global revision. Local page observations must
  reach every consuming content owner before those guards can be safely removed.
  This completes an existing responsibility; it does not add another content cache.
- **Destination connection:** spatially selected draws need valid resident content
  after edits and on newly exposed views. Local content validity is independent of
  raster residency and advances that prerequisite. Native demand remains passive,
  caller-driven and exact-camera. Before enabling asynchronous camera handoff,
  resolve the documented no-ready-image/overlay/picking constraint; live acceptance
  still requires staging/install/game authorization, not further timer plumbing.
- **Architectural acceptance:** preserve exact independent redraws for local/distant
  terrain and river edits, reversals, wrapped inputs, zoom and reset. Observe all
  corridor and height inputs, preserve immutable borrowed-page lifetimes, and bound
  new dependency storage under existing owners and >=512 MiB sampled headroom.
  Keep conservative global validity wherever the dependency audit is incomplete.
- **Performance acceptance and stop:** use the existing small edit oracle first,
  then the fixed dense edit sequence and one unaffected navigation comparison if
  useful. Retain only complete useful reuse without material whole-workload or
  capacity regression. Reject removal of required guards or metadata/validation
  overhead that erases the benefit; do not chase cache sizes or restart baselines.

### Causal breakdown — retained measurement capability, completed diagnosis

The common-depth foundation remains; both coherent animation consumers were removed
from the runtime after exact but slower results. Their source and evidence are kept
only under ignored build evidence. No coherent-pass performance gain is retained.

The optional query owner now separates background preparation/import, receiver
shadows, body/shadow drawing, finishing and staging copy without stage waits. The
existing analyzer aligns GPU/CPU records by request and keeps their overlap explicit.
The VM's D3D timestamp results failed a decisive validity probe: repeated reads of
the same completed query change values, and reversing read order reverses the
reported event order. Both timestamp owners now reject mutable results. Historical
microsecond `valid=1` GPU records are not calibrated GPU execution evidence; CPU
completion measurements remain valid. No invalid GPU durations are treated as zero.

The existing dense 100x100 synthetic map, 2240x1192, waves/reflections disabled,
ran four serial one-case cumulative ablations (14 navigation requests each).
Seven matching resident revisits have identical backdrop hit counts (66–72), no
backdrop misses and no static draw work, which isolates animation more usefully
than the mixed preparation sequence. Values below are means in milliseconds.

| Cumulative work omitted | Whole navigation sequence | Animation on resident revisits | CPU completion wait on those revisits |
| --- | ---: | ---: | ---: |
| None | 156.54 | 49.84 | 32.99 |
| Body/shadow draws | 175.79 | 46.06 | 29.82 |
| Also finishing | 133.41 | 29.24 | 14.64 |
| Also backdrop restoration | 126.51 | 21.78 | 7.45 |

Conditional resident animation reductions are about 3.8 ms for body/shadow draws
(noisy), 16.8 ms for finishing and 7.5 ms for backdrop restoration. CPU bitmap copy
is 0.9–1.0 ms. The last 7.45 ms wait still includes staging/clear/driver/queue effects;
it is not isolated transfer time. These reductions are not additive pure GPU busy
times. Mixed whole-sequence totals vary with preparation, and these intentionally
incorrect ablations are never visual or performance acceptance. The unablated
instrumented-build control preserves all saved shared-depth images exactly.
The local finishing implementation above passed its own whole-workload comparison.

This evidence selects the missing responsibility: damage propagation currently
stops before finishing, repeatedly processing unchanged static content. It does
not justify another round of body instancing or a new backdrop cache. Query lifetime,
pressure/error/disjoint/mutable behavior and analysis alignment/overlap are covered
by executable tests (18 focused tests pass). Current focused resource integration
also passes 254 tests (one existing skip) and exact animation/scroll/removal replay.
Pre-finishing implementation: `eada70a29a31ce85af742642e26ad07f891d22adccad4f76ac10f46f9675d659`;
DLL: `2266b7c464fca76049ee3be3cfd13138039c7c8e82b432e0ece70ad9bfc01e48`.
No staging, installation, injected compilation or live-game claim. Evidence and raw timing limitations:
`Renderer/native/build/coherent-resource-pass-20260913/causal-breakdown.json`,
`timestamp-probe-immutable.log`, and the four `resource-causal-*` run directories.

The first consumer combined up to four horizontal cells in 16.2 MiB scratch.
Exact MSAA import passed; assigning each cell its own interior fixed a one-pixel
seam difference, after which all six boundary redraws and dense saved images were
exact. Whole navigation was 157.79 / 154.06 ms versus a fresh common-depth control
at 154.60 / 147.33 ms. It was rejected as a speedup. The bounded revision used complete
rectangles in both dimensions and imported directly from cached textures, removing
the redundant intermediate cell copies. This changes the consumer implementation,
not correctness thresholds or the architectural destination. Evidence lives under
`Renderer/native/build/coherent-resource-pass-20260913/`.

### Common scene-depth prerequisite — retained

Current map/resource producers now use one explicit scene-depth basis, independent
of raster placement. Nearest-4096 origin changes retire incompatible backdrops;
D24/MSAA4, material order, depth writes, normals and color reconstruction remain.
The user authorized small reviewed depth-rounding differences on 2026-09-13.
The production GPU witness proves compatible regional producers and one common
consumer, including front/behind and identical coplanar order across seven origins.
Focused integration passes 254 tests (one existing skip) and resource playback.
Candidate identity: `374a1f3e7d99e8518745e6998c0bf95405ae6e5aa19fc90c066fc07a49621bae`;
DLL: `7451d4530befb3b25218534824246d7732de824cf7cfa33af55400605e60b484`.

Dense two-case navigation averages 152.50 / 160.17 ms, consistent with preserved
155.80 / 169.75 ms controls but not a fresh interleaved speedup claim. All saved
dense frames preserve the reviewed 25 changed pixels, maximum channel delta 3;
revisits are exact and minimum sampled contiguous headroom is 1701.93 MiB.
The 640x480 zoom warmup is exact at 128/160; 192 differs by 27 pixels (max 16).
Its wrapper detected shader-cache warmup, so it is not timing acceptance.
This prerequisite is retained for the immediate coherent consumer, without an
independent speedup claim. Animation completion remains about 85–94 ms, including
GPU/readback wait whose pure transfer portion has not been isolated.
Evidence: `Renderer/native/build/scene-depth-pass-20260913/`,
`scene-depth-common-navigation-candidate2-20260913`, and
`scene-depth-common-zoom-20260913` beneath `Renderer/native/build/`.
No staging, installation, injected compilation or live-game acceptance occurred.

### Preserved scene-depth implementation decision

The most consequential barrier remains repeated regional dynamic passes. Their
retained static color and D24/MSAA4 depth use guarded region projections. The
larger-pass probe reduced submission from 26–33 ms to 2–6 ms but failed correctness.
The witnesses below now show why preserving every old depth-rounding outcome is
not a simple coordinate conversion: different surfaces can have identical stored
D24 values yet require different values after a common-origin rasterization.

Previously selected task: complete an explicit common scene-depth contract preserving existing material/depth-write rules
for existing static map and resource body/shadow producers through the existing
viewport settings, pass states and backdrop owner. On 2026-09-13 the user approved
allowing the small reviewed depth-rounding differences while preserving visual
occlusion/layering and native ownership. This authorizes the architectural contract
change, not staging, reference replacement or game launch. The common producer
basis is being revised under that authorization; no post-raster import or optional
legacy-depth path is added. The reviewed comparison is
`Renderer/native/build/scene-depth-20260913/depth-contract-review.png`.

The pass-owner inspection narrows the implementation boundary: the current farm
compiler distinguishes base, crop and building components, then merges them into
`farm_vertices`; `submit_geometry` draws that entire layer with depth writes enabled.
Disabling depth writes for all farms would therefore also affect buildings and is
not a justified fix. The existing natural-decal pass already uses a depth-tested,
non-writing state. Any explicit coplanar pass must preserve component semantics
and opaque-body occlusion through the existing compiler/draw owners, rather than
assigning behavior from texture indices or treating every farm component as flat.
A candidate separated only horizontal crop assets into a non-writing ground pass,
preserving building depth writes. It passed the boundary witness but changed 60,617
dense pixels (maximum channel delta 68), well beyond the reviewed rounding change.
That split and its dedicated test extension were removed. Existing farm/material
depth-write rules are retained; the common-depth basis remains the selected
prerequisite. Evidence is under `Renderer/native/build/scene-depth-pass-20260913/`.

- **Capability afterward:** retained static depth and selected dynamic instances
  can participate in a coherent view/band pass with explicit compatible depth and
  overlap rules. This enables replacing repeated regional resource rendering.
- **Work and limitation removed:** region-dependent numerical tie-breaking stops
  defining the architecture. The subsequent coherent pass, rather than this
  prerequisite alone, must remove repeated body/shadow submission and finishing.
  Use existing owners and pass states; do not add caches or preserve another
  optional rendering path merely to reproduce rounding artifacts.
- **Connection to the destination:** the coherent consumer is the immediate next
  capability if this contract passes; general spatial selection and compatible
  batching can feed it. New views must render resident world content efficiently.
  Native consumption remains caller-driven and exact-camera compatible. The
  strategic live checkpoint still needs separate staging/install/game authority.
- **Architectural acceptance:** preserve existing material order and depth-write contracts,
  correct overhang/body/shadow occlusion and stable camera/origin behavior. Preserve
  D24/MSAA4, source normals, color reconstruction, ownership and visibility. Compare
  retained operation against independent full redraws under the new contract,
  including camera translation, wrap, zoom, removal and reset. Preserve and report
  old/new image differences separately; do not silently relax an oracle or count
  the permission as acceptance of untested future changes. Account for
  metadata and transient targets within existing budgets and >=512 MiB headroom.
- **Performance acceptance and stop:** a correct bounded prerequisite may be
  retained without a large immediate speedup only if it avoids material regression
  and demonstrates the common consumer dependency on the GPU. Reject unhelpful
  complexity. Stop origin/format/offset tuning: the preserved witnesses already
  establish the old rounding constraint. Test the complete bounded
  depth/overlap contract before allocating the larger consumer target.

### Post-raster depth import — rejected at the small witness

The alternative preserves static color/rasterization, imports each D24/MSAA sample
into a second depth target, and draws the identical coplanar captured farm quad
through the proposed common consumer. The unchanged-origin import is exact.
Integer-code translation fails at -128 (15 pixels) and +2,048 (292 pixels);
normalized floating translation also fails at ordinary offsets, including +636
(471 pixels). These are diagnostic supersampled pixels with flat colors, not
claims of full-scene visual difference. No import code entered the renderer.

A further GPU witness reads actual per-sample depth codes from two differently
tessellated captured farm surfaces. At **11 identical sample positions**, both
surfaces have the same old D24 code but different common-origin codes. For example,
old `8249803` becomes `7598540` for one surface and `7598539` for the other at the
same pixel/sample and +636 offset. Thus the stored depth value, origin and sample
position do not contain enough information for a unique exact conversion to the
newly rasterized depth. A depth-only rebase cannot generally reconstruct it.
This is a concrete architectural tradeoff, not a reason to try more offset values.

`depth-import.cpp/.log`, `depth-ambiguity.cpp/.log` and `decision.json` under
`Renderer/native/build/scene-depth-20260913/` preserve the executable evidence.
The permission checkpoint above is required by the current exact-output contract;
no visual acceptance, performance acceptance, staging or live integration is claimed.

### Common floating-point depth basis — rejected and removed

The candidate separated depth translation from raster XY across static and dynamic
producers, with bounded world-origin metadata in the existing backdrop owner.
Builds and two behavioral contracts passed. A D24/MSAA4 witness consumed static
depth from two regional projections in one larger dynamic pass with exact simple
occlusion/coplanar ordering; its intentionally incompatible basis changed 4,096
pixels. Five small native publications and six guarded-boundary redraws also
matched the accepted renderer. These simple witnesses did not cover differently
tessellated overlapping farm surfaces.

The dense candidate passed same-build revisits but differed from the accepted
renderer at **25 pixels, maximum channel delta 3**. Its 153.594 ms mean is not a
performance acceptance result. A temporary capture then extracted the actual
retained buffers/settings in the region containing a differing pixel. The capture
hook was removed after that one diagnostic; its timings are excluded.

A small GPU replay of that captured farm chunk, using the production vertex
adapter and flat diagnostic colors, reproduces **44 changed supersampled pixels**
when only the depth offset changes by 636 pixels. Whole-region shifts of +/-128
also change 27 pixels; shifts of 512, 1,024, 2,048 and 4,096 fail too. The same
geometry and order therefore select different overlapping layers after pre-raster
depth translation. This establishes the precision-sensitive ordering dependency;
the exact internal rounding stage has not been isolated. Changing the origin's
phase/granularity is not a reliable fix, and no tolerance or appearance change is
accepted.

The implementation and its tests are archived under
`Renderer/native/build/scene-depth-20260913/rejected-floating-basis-source/`.
`decision.json`, `captured-region.bin`, `pixel-triangles.json` and the two
`farm-depth*.log` witnesses preserve the diagnosis. The original source snapshots
were restored and dependent shaders regenerated through the preparation guard.
Current compiled inputs and implementation identity exactly match the previous
passing resource Integration receipt; no redundant baseline campaign was run.
Shared resource geometry and all previously validated improvements remain intact.
Post-raster import was then tested and rejected as recorded above. The single next
task requests an explicit stable depth/overlap contract rather than another attempt
to preserve every legacy rounding outcome.

### Ordered resource instancing — rejected and removed

The preflight found real compatible batches: 310 selected draws formed 135
consecutive runs in the initial dense view; after movement, 292 draws formed 145
runs. Free reordering would not lower those counts further in these views. The
candidate implemented one shared typed pose/placement upload, one selected instance
stream, and indexed instanced draws with preserved order and independent phase.
It removed per-instance constant-buffer allocation/binding and added no pose cache.

**Correctness/resource evidence:** isolated builds and two behavioral pose/batch
contracts pass, including denied-growth preservation. The small native witness
passes five exact publications, and its images match the accepted CPU renderer.
Supported zoom images at 128/160/192 are also exact against the existing independent
control; that zoom invocation was a shader-cache warmup, not acceptance timing.
All four dense timed cases preserve saved images and exact revisits, inputs,
binaries, ownership/accounting and exclusive-GPU checks. Candidate minimum sampled
contiguous headroom was 1,688.6 MiB; the 32 MiB resource limit was unchanged and
included upload-growth transients.

**Whole-workload decision:** candidate means were **175.920 and 161.390 ms**;
the intervening matched controls were **155.804 and 169.746 ms**. These are mixed
results, not a repeatable improvement. Fewer calls did not establish faster dense
navigation, so the extra upload/submission code and its dedicated test were removed.
The previous shared mesh/instance representation remains intact. No broader
integration campaign or native installation was run for the rejected candidate.

Evidence and exact source are preserved under
`Renderer/native/build/resource-instance-batches-20260913/` (`comparison.json`,
`opportunity.json`, small/zoom comparisons and `rejected-source/`). The first control
invocation detected changed shader-cache inputs and stopped before timed cases;
it is excluded. The single next task above addresses the pass/depth boundary,
not further tuning of pose buffers, batch ordering, or raster-cache policy.

### Shared resource geometry — structurally retained

The retained implementation draws resident source meshes using per-instance
bone/placement constants in vertex shaders. It removes per-occurrence CPU skinning,
CPU camera projection and full body/shadow vertex uploads. Both existing passes
share the source and pose contract; legacy visual profiles keep their established
CPU path. No optional cache, new queue or presenter is introduced.

An initial dense comparison found two changed wheat-edge pixels. Exact CPU bounds
did not remove them. A bounded GPU probe isolated one-bit world-coordinate rounding;
precise world arithmetic restored exact dense images without changing the shared
compiler or relaxing equality. The temporary CPU-bounds diagnostic was removed.
The retained sources and comparison receipt are under
`Renderer/native/build/resource-instances-20260913/retained-decision.json`.

**Architectural decision:** retain the shared mesh/instance foundation as the user
requested. It exposes reusable mesh and independent instance inputs that the CPU
screen-space streams did not. Those inputs can feed a coherent scene pass after
the selected common-depth contract is established; the unsuccessful regional
instance-submission variant above is not required to keep this foundation.
It is not a complete world database, general batching, or a navigation gate pass.

**Performance evidence:** corrected candidate means were **154.843 and 168.703 ms**;
the intervening two `assets_loaded` CPU control cases were **187.090 and
159.364 ms**. All four 14-offset cases preserve saved images, inputs,
ownership/accounting and exclusive-GPU checks. Results are mixed: no reliable
whole-workload speedup or material regression is established. The small witness's
recurring uploads fall from 1,173,960 to 22,288 bytes; that is removed transfer work,
not a claimed frame-time improvement. Dense sampled resource storage peaks at
420,664 bytes versus 12,120,192 in the CPU control, under the unchanged 32 MiB cap;
unchanged CPU asset payloads and other renderer owners remain separately accounted.
Minimum candidate sampled contiguous headroom is **1,707.8 MiB**.

**Dominant remaining cost:** animation composition averages about 85–94 ms in the
two candidate runs, including repeated regional submissions and GPU/readback wait.
Pose preparation drops to about 6.3–6.4 ms, but that alone does not make navigation
fast. The selected instance-submission capability must demonstrate useful end-to-end
consequences; the below-100 ms target and live presentation remain unmet.

**Rejected alternative:** compute-once skinning into a shared posed buffer compiled
and passed CPU pose contracts, but failed native correctness: the initial small
image omitted resources (11,723 changed pixels versus the CPU control), and the
dense exact revisit failed at offset step 4. Its timing is excluded. Both compute
shader and UAV storage were removed from active code; source/evidence is preserved
under `compute-raw-rejected-source/` and
`Renderer/native/build/resource-compute-navigation-candidate1-20260913/`.
Do not keep or revisit that extra machinery on a speculative benefit.

**Verification:** 133 earlier resource-category tests (one existing skip), nine
focused pose/bounds checks and the five-publication small native witness pass.
The corrected dense images are exact against the accepted CPU control. Current production verification passes 252 focused tests (one existing skip),
six temporal frames, zoom-return, exact cold scroll and removal comparisons, with
zero fallback. Receipt: `Renderer/lab/out/integration/resources.json`. The required
category build exposed an existing batch-variable/toolchain discovery error;
`BUILD.bat` and the Lab preview build now use the already verified benchmark
discovery, including preview Visual Studio installations. Production shader/source
identity is verified; no staging, installation, reference replacement or Civ III
launch occurred. Additional supported-zoom independent witnesses remain necessary
when changing the selected instance submission contract.

### Native request identity — structurally retained

The optional `c3x_renderer_render_view` ordinary-demand export uses the existing
versioned camera request, worker queue and publication owner. Injected capture now
supplies scenario lifetime, native viewer, observed visibility and topology scene
revision; ordered local tile/object/anchor data still participates in exact request
matching. Changing any identity rejects old work. Identical requests consume/join
existing work rather than cancelling and executing it again. Same-view ambient
completion remains passive until Civ III next calls. Legacy API-17 callers and DLLs
remain compatible; no second presenter, completion callback or redraw hook exists.

Architectural acceptance passes: seven executable native-identity/publication tests
cover epoch changes, exact adoption, reset/failure/unit takeover, visibility-only
edits, mixed-viewer rejection and allocation failure. A final review caught a partial
allocation hazard when shrinking the world observations; both owners now commit
only after allocation succeeds, with regression coverage for failure at either
allocation followed by returning to the old map. Another 38 bridge/fixture tests
pass. The final approved injected compile and the isolated normal-tier native DLL/
preview build pass. The restored pre-change renderer also passed 73 focused checks.

The native `native-view-ambient-small-20260913` witness passes all five independent
exact publications with nonzero epochs and verified source/binary/input identity.
Three stationary calls took 0.501, 0.228 and 0.210 ms; their publication completion
polls observed 32, 15 and 31 ms. An explicitly queued current request was adopted
by ordinary demand in 15.143 ms (15.253 ms including begin), with no second render.
The changed-camera request still waited 640.167 ms for exact output. These sparse
640x480 standalone samples validate the contract, not dense navigation or native
frame rate. Minimum sampled contiguous headroom was 2038.8 MiB; no fallback or
recovery occurred. Tests/compile hashes are under `native-view-identity-20260913`,
and the complete native build is `native-view-complete-build-20260913`, all beneath
`Renderer/native/build/`.

The native visibility observation adds eight bytes per parity tile to the existing
world scan: 40,000 bytes for 100x100, at most 16 MiB at the supported 2048x2048 cap.
Resize temporarily overlaps old/new visibility allocations; neither publication
nor geometry budgets increase. When full-world topology capture is disabled,
visibility epoch remains zero and exact ordered local visibility still governs
reuse. This is not complete world appearance capture or measured live capture cost.

**Remaining native limitation and checkpoint:** same-view asynchronous ambient
rendering is still the existing opt-in mode. A new camera without compatible output
must finish exactly before native overlays/picking proceed. Current source is ready
for a strategic staged/installed game check of scenario/viewer/visibility changes,
stationary ambient demand, camera/zoom/overlay/picking and unit takeover, but no
staging, installation or game launch was performed or authorized here. That check
is required before enabling/claiming ordinary native async presentation or native
frame-rate gains; general asynchronous camera changes cannot bypass the exact-view
constraint. Independent resident-content work can proceed while it remains pending.
No new executable patch-table entry is required.

### Fixed-topology terrain batch experiment — rejected and restored

The bounded candidate packed immutable natural terrain vertices into pages keyed
by exact index bytes/format and vertex size. Consecutive eligible occurrences used
indexed instanced draws with their original projection; shadow/reflection consumers
used the same allocations at byte offsets. Pages were owned by compiled-content
leases, with slack/index/instance storage charged to the unchanged geometry budget.
This was a real submission mechanism, not the earlier bounds-only index experiment.

Architectural checks passed at the small scope: 144 selected terrain occurrences
became 54 submissions in the initial boundary frame, preserving alpha/triangle
order. `terrain-batch-defaults-boundary-20260912` passed all six independent redraws
and exactly matched the accepted occurrence renderer's saved images with full
input/source/binary verification. The page ownership/failure/bounds test and 251
selected resource/integration tests passed (one existing skip). These do not certify
broader zoom, edit or live-game behavior; those campaigns were not earned by the
performance result.

The deciding dense 100x100, 14-offset navigation cases used the same production
defaults, normal budgets, `assets_loaded` reset and exclusive-GPU watchdog:

| Order | Control whole request | Candidate whole request |
| --- | ---: | ---: |
| control, candidate | 163.522 ms | 164.519 ms |
| candidate, control | 160.968 ms | 200.267 ms |

All 56 playback endpoints passed ownership/accounting checks; saved images, runtime
inputs and request manifests matched. All watchdogs were clear. Candidate contiguous
headroom remained at least 1559.098 MiB (controls at least 1701.105 MiB). Initial
preparation was 13.414/13.566 seconds control and 13.574/13.701 seconds candidate.
The second candidate had broad phase variation; preserve it without claiming a
proven cause or excluding it. Neither pair demonstrated useful whole-request gain.
The first pair's main draw phase was 6.646/6.760 ms; animation composition remained
91.755/92.197 ms. Fewer terrain submissions did not remove the dominant workload.

Reject this representation rather than retaining its paging/raw-vertex complexity
on an unspecified promise of future gains. The compact occurrence records and all
preceding accepted improvements remain; no live batch capability is claimed.
The candidate sources/shaders and comparison receipt are preserved under
`Renderer/native/build/terrain-batch-navigation-candidate2-20260912/`.
The accepted pre-batch runtime, shadow owner and natural shader runtime were restored.
Obsolete batch implementation/tests were removed from the active source tree.
Three stale native source assertions were updated to name the already accepted
captured-appearance/content owners; their executable owner tests remain.

Earlier setup receipts are preserved: the first shader include form failed the
existing flattened-source loader; a warm-cache run cannot pass input stability.
An initial replay command omitted inherited production defaults and produced five
one-level color differences in both control and candidate; restoring the documented
switches produced exact accepted images. Neither setup issue is performance evidence.
The final candidate generated its batch vertex entry separately so existing shaders
remained unchanged. No art/reference acceptance, staging, install or game launch occurred.

### Resident occurrence draw records — structurally retained

The user explicitly directed retaining this step if it is a dependency for
architectural pieces likely to produce decisive performance gains. Retain the
**revised borrowed-reference implementation** on that basis: general selected
draw lists need shared content handles and occurrence projection across drawing,
shadow selection and region validity without cloning mesh/material ownership. Those
consumers now share that interface. The rejected terrain page allocator was one
consumer of it, not its only architectural purpose. No speedup or
completed general spatial-selection capability is claimed for the representation.

`occurrence-draw-reference-complete-build-20260912` is the accepted build. Static
view assembly and pixel prefetch now keep source references plus projected bounds,
translation and natural projection; immutable mesh/material metadata stays in
`CachedTileGeometry`. Static drawing, receiver/caster selection and region keys use
one read adapter. Active epoch protection preserves source lifetimes through
compilation/prefetch, with view clearing before eviction/reset destruction. The
view takes no COM/material references; dynamic poses/waves keep their existing
owners. Read traversal copies two pointers, not the full projection record.

Focused owner/mesh/region tests pass (30), including borrowed versus independently
traversed region keys, local dependencies, order and supported projected bounds.
The assembly test works with non-copyable source chunks and verifies unchanged
resource reference counts. A focused caster test additionally matches borrowed
and owned traversal for every wrapping combination with different occurrence
projection; it passes. Resource category tests pass (132, one existing skip).
The original build caught one remaining owned-container traversal; the reference
revision required an explicit array-reference return for the existing MSVC mode.
Both were corrected before the accepted build, with unchanged verified sources.

`occurrence-reference-boundary-20260912` passes all six independent small redraw
checks. `occurrence-reference-zoom-small-20260912/comparison.json` records exact
128/192/160 repeats and saved-image parity with accepted content bindings, using
identical runtime inputs. Profiled static-view metadata peaks at 80,160 bytes in
that 640x480 fixture and returns to the fixed 564-byte layer-array metadata after
reset. Sampled contiguous headroom exceeds 2030 MiB. The source enforces occurrence
records at most half the owned chunk size; allocator/driver transients are not
fully measured. `occurrence-reference-edit-small-20260912` passes four distant/
visible edits and reversals against independent full redraws. The visible calls
remain 1260.039/1239.929 ms; these single-arm correctness timings are not a speedup.

`occurrence-reference-navigation-candidate3-20260912/comparison.json` preserves
three matched 14-offset cases per arm, first control/candidate then candidate/
control twice, with `assets_loaded` reset and identical initial preparation. The
2240x1192, width-128, waves/reflections-off dense 100x100 synthetic world retains
1921 initial occurrences, 5 cities, 349 roads, 80 rails, 344 improvements and 150
resources. All 84 playback endpoints pass ownership/accounting checks; saved images
match exactly. Full input/source/binary verification and all watchdogs pass. Dense
saved-image equality is not an independent full dense redraw oracle.

| Mean whole navigation request | Case 1 (ms) | Case 2 (ms) | Case 3 (ms) |
| --- | --- | --- | --- |
| Accepted content-binding control | 157.147 | 159.723 | 158.803 |
| Borrowed occurrence records | 154.171 | 168.342 | 156.666 |

**Architectural result:** retained as the direct input to the selected batch pass.
**Performance result:** inconclusive for speedup; aggregate means are 158.558 ms
control / 159.726 ms candidate (about 0.7% slower). The slower second candidate
repeat remains in the evidence; it did not recur in the deciding third pair.
This closes the comparison without extending a tuning campaign. Initial preparation
is 13.035/13.228/13.235 seconds control and 13.167/13.169/13.424 seconds candidate.
Animation composition remains dominant (86.716/91.570/86.790 ms candidate);
geometry is 31.499/37.168/33.520 ms. Headroom across the six runs stays above
1701 MiB at sampled host boundaries. The below-100 ms navigation gate is unmet.

The first implementation is rejected: its read adapter returned projection records
by value on every traversal. Its exact small boundary/zoom output did not justify
159.495/158.517 ms controls versus 174.732/176.786 ms candidates. Draw means rose
from 6.524/6.547 to 8.343/8.581 ms, and animation composition from 86.425/86.436 to
95.794/95.024 ms. All 56 endpoints and verification checks passed with no watchdog
conflicts. The first source remains under
`occurrence-draw-complete-build-20260912/first-candidate-source/`; its comparison
is `occurrence-draw-navigation-candidate2-20260912/comparison.json`. The revision
removed the added per-traversal copying; it is not claimed to explain every timing
variation. No optional switch, GPU budget, shader appearance, native hook or ownership
category changed. No staging, installation, game launch or visual promotion occurred.

### Resident compiled-content bindings — structurally retained

`content-binding-complete-build-20260912` builds the retained implementation. The
existing mesh cache remains the sole buffer owner. A slot table, bounded by that
cache's existing entry limit, supplies non-owning generation handles; eviction/reset
invalidate handles without retaining buffers. World records, cached-view arrays,
projection entries and pixel preparation consume these handles. World-bound reuse
still checks the exact compiled signature and every existing dependency. The keyed
owner lookup remains for a missing binding/new context. No mode or GPU budget grew.

Host owner/mesh tests pass (20 tests), including stale slots after eviction/reuse,
reset, failed capacity admission, no lifetime extension, real cache eviction,
viewport restoration and prefetch rejection. Resource category tests pass (132,
one existing skip). The initial build found a remaining iterator-style prefetch
access; it was converted before the successful build.

`content-binding-boundary-20260912` passes all six small independent full-redraw
resource/ownership/clock/removal checks. `content-binding-zoom-small-20260912/comparison.json`
records two cycles at widths 128/192/160: all repeat comparisons and saved images
against the preceding accepted control are exact. These are profiled 640x480
correctness checks, not a zoom speedup claim. The handle table reaches 25,584 bytes
there and returns to zero on reset; sampled contiguous headroom exceeds 2036 MiB.

`content-binding-edit-small-20260912` additionally passes all four distant/visible
terrain edits and reversals against independent full redraws at 640x480, with
unchanged inputs/sources/binaries and no watchdog conflicts. This closes local
appearance/dependency invalidation coverage for the binding change; its single-arm
timings do not establish an edit speedup.

`content-binding-navigation-candidate-20260912/comparison.json` records two dense
14-offset cases per arm, controls then candidates, each with `assets_loaded` reset
and identical initial preparation. The 100x100 synthetic world supplies 1921 initial
captured occurrences, 5 cities, 349 roads, 80 rails, 344 improvements and 150 resources
at 2240x1192, width 128, waves/reflections off. All saved images match across arms;
all 56 playback endpoints pass ownership/accounting checks. Dense saved-image parity
is not an independent full dense redraw oracle.

| Mean whole navigation request | Case 1 (ms) | Case 2 (ms) |
| --- | --- | --- |
| Preceding accepted control | 165.582 | 183.783 |
| Resident content bindings | 160.334 | 183.937 |

**Architectural acceptance:** retained. Lifetime-checked content association replaces
key/version rediscovery in real consumers without another buffer owner. **Performance
acceptance:** no material regression in these repeats, but no reliable speedup or
below-100 ms pass. Initial preparation is separately 14.230/13.449 seconds control
and 13.002/13.205 seconds candidate. Animation composition remains dominant at
88.561/99.504 ms candidate; geometry is 34.071/42.459 ms. This does not select another
animation helper as the next task.

Up to 2425 world records occupy 2,635,448 tracked bytes versus 2,596,648 control.
Host-boundary contiguous headroom stays above 1699 MiB across both arms; internal
transient/driver residency and dense handle-table peaks remain unmeasured. Existing
GPU budgets are unchanged. Inputs/request manifests match; source differences are
the renderer and the captured-scene/resident-content headers. Full source/input/
binary verification passes, and both dense watchdogs report no competing GPU/build
process. Small boundary/zoom runs are one-shot without watchdog coverage.

Shared content is still copied/projected into per-view chunks; complete appearance
capture, general spatial draw lists, compatible batching and native presented-frame
performance remain unfinished. No staging, installation, game launch, injected
changes, visual promotion or live-game performance claim occurred.

### Persistent captured appearance — structurally retained

`SceneTopology` now extends the worker-owned `CapturedScene` record table instead
of owning six rebuilt maps and borrowed appearance pointers. Records keep exact
current occurrence data, canonical static appearance and a local revision. Camera,
clock, visibility and unit-body changes do not change static appearance identity;
actual tile-attached changes do. Lightweight halos preserve remembered full
appearance but retain their own current lookup data. The existing last-ordered
canonical occurrence rule is unchanged. Absent records cannot serve current
lookups, and an incomplete/cancelled update exposes no partial current lookup.

The bounded observed subset retains at most 8192 records, matching the API's
maximum occurrence count. Capacity pressure evicts absent records only, after
protecting every current coordinate regardless of traversal order. Eviction and
reinsertion receive a new revision within the owner lifetime. World basis changes
clear remembered records; the existing geometry/reset lifecycle still clears this
CPU owner too. Payload is statically bounded below 16 MiB; tracked bytes include
estimated node overhead and actual bucket count, with allocator/transient limits
reported separately. This is not complete full-world capture, durable object/action
identity, GPU-reset-independent CPU retention, or general spatial selection.

The compiler's terrain, river/route, natural/city-exclusion and dependency lookups
now consume this single owner. Combined neighborhood queries perform one lookup
instead of separate ground/surface/relief lookups. The current geometry compiler,
GPU caches, raster reuse, pass ordering and public capture ABI are preserved.
There is no new opt-in mode or parallel lookup implementation.

Owner and retained mesh tests pass (19 tests), covering owned-copy lifetime,
revision stability, wrapped ordering, full/halo precedence, removal, cancellation,
world replacement, bounded admission/eviction and existing mesh dependencies. The
old mesh test fixture's map lookup was adapted to the new record lookup without
removing dependency assertions. `renderer.py test resources` passes 132 tests with
one existing skip. The new owner tests are included in ordinary integration checks.

`captured-scene-complete-build-20260912` built the measured candidate after the
initial build identified one remaining forest/city lookup on the deleted map;
that lookup was converted. `captured-scene-boundary-20260912` passes all six exact
resource/ownership/clock/removal checks. `captured-scene-edit-small-20260912` passes
four distant/visible edits and reversals against independent full redraws. The
640x480 edit calls were 20.496/20.256 ms distant and 1207.540/1251.307 ms visible;
these single-arm figures are correctness context, not an edit speedup claim.

`captured-scene-navigation-candidate-20260912/comparison.json` records two dense
14-offset cases per arm, controls then candidates, each with `assets_loaded`
reset and identical initial preparation policy. The complete 100x100 world has
1921 initial captured occurrences, 5 cities, 349 roads, 80 rails, 344 improvements
and 150 resources at 2240x1192, width 128, waves/reflections off. All saved images
match across arms; all 56 playback endpoints pass ownership/accounting checks.
Small independent redraw checks establish correctness beyond revisit equality
at their small scope; dense saved-image parity is not a full dense redraw oracle.

| Mean whole navigation request | Case 1 (ms) | Case 2 (ms) |
| --- | --- | --- |
| Control | 158.736 | 183.991 |
| Persistent scene owner | 168.875 | 154.988 |

Variation is larger than a reliable speedup claim. Retain the structural ownership
replacement with no material regression established across these repeated cases;
the below-100 ms gate remains unmet. Animation composition remains the largest
measured component (candidate means 89.605/83.218 ms); geometry means are
38.600/33.322 ms. The next task follows the missing content association, not the
largest helper. Up to 2425 scene records occupy 2596648 tracked bytes (about
2.5 MiB), and sampled contiguous headroom stays at least 1696.2 MiB across both
arms. Internal transient/driver residency is unmeasured. Inputs/request manifests
match; source differences are the renderer, captured-scene header and one included
geometry lookup. Full source/input/binary verification passes and the session
watchdog observes no competing GPU/compiler process. The small six-case boundary
is one-shot without watchdog coverage; the edit/dense cases use it.

The final `captured-scene-documented-build-20260912` only clarifies the header's
bucket-accounting comment relative to the measured candidate; runtime code is
unchanged. No staging, installation, game launch, injected changes, visual promotion
or native presented-frame claim occurred.

The pull contract is resolved: completion never requests a native redraw. Native
map/viewer/visibility epochs and exact displayed-view compatibility remain required
before enabling general asynchronous camera handoff in Civ III. Until that bridge
is implemented, a missing exact camera blocks for its result; old-view pixels do
not become acceptable because preparation is asynchronous. This boundary must be
resolved at the native handoff implementation, before its strategic live checkpoint,
not inferred from upstream scene speedups or deferred behind a frame-rate claim.

### Exact queued-request consumption — structurally retained

`RendererWorker::render` now joins an identical pending/active camera request or
consumes its completed publication through `camera_poll_locked`. It reuses the
existing immutable snapshots and condition variable. Legacy calls adopt only zero
caller-owned lifecycle epochs; the optional ambient path also rejects ambiguous
nonzero epochs. Different camera, clock, visibility, order or topology still takes
the existing synchronous path. An adopted failure returns its error without
publishing an older front as success. No owner, allocation, byte cap, ABI layout,
mode default or injected code changed. API comments document passive consumption.

The host C++ worker tests establish one execution for active, ready and pending-
behind-obsolete-copy requests, exact pixels/ownership, all four epoch rejections,
changed input rejection, failure atomicity/recovery and preserved reset/unit
interruption. Publication, navigation fixture and endpoint analysis checks pass
(20 tests). The actual worker test uses host clang; the build and replay below use
Windows x86 D3D11. No injected compile is required for these Renderer-only changes.

`pull-adoption-contract-build-20260912` is the successful current build.
`pull-adoption-ambient-small2-20260912/comparison.json` preserves two matched
640x480, width-128 boundary pairs in candidate/control then control/candidate order.
Each fresh process runs the same three stationary clocks, changed camera and queued
return to resident home content, with independent synchronous references. All 20
publication checks are exact with no fallback or recovery. Candidate traces show
the queued request finishing through the camera owner and being adopted; controls
replace it with an ordinary synchronous job. The deterministic host test, rather
than an assumed native scheduling race, proves avoidance of duplicate execution.

| Endpoint | Control repeats (ms) | Candidate repeats (ms) |
| --- | --- | --- |
| Queue begin through ordinary exact return | 15.244 / 15.111 | 14.731 / 14.057 |
| Changed camera through exact return | 652.196 / 629.174 | 630.295 / 626.545 |
| Maximum stationary ordinary call | 0.549 / 0.367 | 0.425 / 0.526 |

This supports structural retention with no material regression in the small
workload, not a meaningful speedup or native cadence claim. Queue timing includes
submission and waiting over a prebuilt immutable capture; live capture and actual
presentation are unmeasured. Sampled contiguous headroom is at least 2031.2 MiB;
internal transient/driver residency is unmeasured. Inputs match across arms, the
only native source difference is `c3x_renderer.cpp`, and all source/input/binary
verification passes. Runs were serial one-shot fixtures without session
exclusivity-watchdog coverage. Control source was restored to the candidate after
each run. Changed-camera reconstruction remains expensive; this handoff does not
solve it. No staging, installation, game launch or live-game acceptance occurred.

### Publication-owned front validity — structurally retained

`PublishedMapFrame::matches_static_view` now checks the front's own exact captured
occurrences, camera/world basis and topology revision before the ambient legacy
path can return it. The existing immutable request comparison separately checks
full topology bytes. An active job matching the caller no longer certifies a front
from a different view. No allocation, budget, native code, shader or mode default
changed for this correction.

The six publication/worker tests pass on the host C++ path. The new mixed-interface
case fails against the previous guard (`!returned.load()`), reproducing old-camera
return while the new camera is in flight. It passes with the correction, along
with passive next-call completion consumption and no repeated execution of an
unchanged clock. The initial test timeout came from reusing the earlier fixture's
two-record array instead of the intended changed tile; the fixture was corrected.

`pull-publication-build-20260912` is the successful isolated native build.
`pull-publication-ambient-small-20260912/comparison.json` compares it with the
accepted control on the existing 640x480, width-128 ambient boundary: three
stationary clocks plus a changed camera, all exact against independent synchronous
references. Runtime inputs match; the only native source difference is this guard;
all run inputs/sources/binaries verify unchanged. Maximum measured stationary
calls were 0.394 ms candidate / 0.359 ms control. Changed-camera exact completion
was 610.993 / 614.073 ms. Sampled contiguous headroom was at least 2034.8 MiB.
One pair supports no material regression at this scope, not a speedup/tail claim.
Both runs were serial one-shot fixtures without session exclusivity-watchdog
coverage; internal transient/driver residency is unmeasured. The earlier attempt
to reuse an old build directly was correctly rejected by source-identity checks;
the valid control was rebuilt/reverified, and the candidate source was restored.
The wrapper only exposes and validates the existing ambient boundary fixture.
No Civ III launch, staging, injected build or native presented-frame claim occurred.

The helper-led natural-mesh task was not implemented. Its final small profiling
probe, `natural-mesh-edit-phase-small-20260912`, passed four exact edit/full-redraw
checks and input verification. Visible edits spent about 357 ms in natural ground
and 355–356 ms in relief, within a 757–758 ms natural phase. This is preserved
attribution, not a task selection. The river-field experiment is closed below.

### Local river-field inputs — rejected after the small exact comparison

`local-river-input-build-20260912` implemented an opt-in dependency snapshot for
the existing 16-page owner, including height callback inputs beyond its nominal
halo. The candidate reserved at most 256 KiB of snapshot payload across the cache;
page ownership and immutable held handles remained bounded. CPU parity covered
distant edits, river removal/return, relief beyond the source halo, coast, wrapping,
owner/dimension changes and reset, alongside the existing 312-sample/LRU witness.

`local-river-edit-small-candidate-20260912/comparison.json` records one matched
640x480 pair over the complete 100x100 world, width 128, fixed clock, waves and
reflections off, with accepted shadow/local-region paths enabled in both arms.
All eight edit results passed independent exact full redraws, ownership and
animation counts. Full input/source/binary verification matched and no competing
GPU/build process was observed. Candidate retained 10/10 pages on distant changes
and 6/10 on visible changes, versus zero for control.

Despite that reuse, mean visible edit latency was 1274.791 ms candidate versus
1256.385 ms control; mean complete four-edit latency was 648.291 / 639.794 ms.
Distant edits were 21.792 / 23.203 ms. These single-pair differences do not establish
a regression or tail behavior, but show no useful small-workload gain and do not
earn dense repetition. Geometry remained about 1.1 seconds on visible edits.
The compiled river-field work was not a material part of that cost in this fixture.
The candidate, its switch and candidate-only tests were removed. Source snapshots
and the worktree patch are preserved under the build's `rejected-source/`;
the accepted shadow/local-region implementation is unchanged. No broad category
or larger GPU campaign was run for this rejected candidate. Host-boundary memory
samples remain in the receipt; internal transient/GPU-query coverage is unmeasured.

### Local region validity — retained after dense exact/performance validation

`local-region-dense-edit-build-20260912` is the validated isolated build. Its
renderer is unchanged from the small fixture; the preview adds host-boundary
address-space samples outside the measured edit interval.

`local-region-edit-dense-candidate2-20260912/comparison.json` records two matched
pairs in alternating arm order: 2240x1192, width 128, fixed clock, waves/reflections
off, normal budgets and the accepted shadow path in both arms. The complete
100x100 world supplies 1,921 captured occurrences, including 5 cities, 349 roads,
80 rails, 344 improvements and 150 resources. Every distant/visible edit and
reversal matches independent full-redraw pixels, ownership and animation counts.
Inputs, sources and binaries match across all four runs; full verification passed
and no competing GPU/build process was observed.

| Mean edit latency | Control repeats (ms) | Candidate repeats (ms) |
| --- | --- | --- |
| Complete four-edit sequence, per edit | 1206.060 / 1181.455 | 1002.073 / 983.028 |
| Distant edits | 465.394 / 424.929 | 102.641 / 115.685 |
| Visible edits | 1946.727 / 1937.980 | 1901.506 / 1850.370 |

Whole-sequence paired savings are 203.987 / 198.427 ms per edit, or 16.9 / 16.8%.
For distant edits, 66 unchanged animation backdrops are reused instead of rebuilt.
Visible edits reuse 21 backdrops and rebuild 45, while both arms rebuild 81 geometry
tiles. These counts explain the measured result; they are not the retention gate.
The minimum sampled contiguous address-space headroom is 1,613.375 MiB. Sampling
is at host request boundaries; internal transient/GPU-query coverage is unmeasured.
The fixed mix of two distant and two visible edits is a diagnostic workload,
not an estimate of their frequency in real gameplay.

`local-region-navigation-candidate-20260912/comparison.json` is one matched
unaffected-navigation regression pair: 193.386 ms control / 193.833 ms candidate
on the 14-offset dense reversal, with identical saved images/exact revisits,
matched inputs/sources/binaries and no observed GPU conflict. Minimum sampled
contiguous headroom is 1,692.859 MiB. This establishes no material regression in
that pair, not a new navigation speedup or a long-tail claim. Current resource
checks pass (132 tests, one existing skip). Retain the local-region implementation
under its existing control; broader coast/river/effect/lifecycle pixel coverage
and native presentation remain unpassed scopes, not implied by flat-land edits.

`local-region-edit-isolated-build-20260912` built the initial opt-in
`C3X_RENDERER_LOCAL_REGION_REVISIONS` candidate. Geometry still validates each
topology change; only the retained-region context substitutes local contributor/
shadow dependency validity for the global topology revision. Device/content,
projection, lighting, wrap and reset contracts remain in place. No cache cap or
native ownership changed. The same context governs animation backdrop reuse.

The existing preview now has a focused topology-edit fixture: one distant and
one visible flat-land edit, each reversed, at fixed view/time, with independent
reset/full-redraw pixels and ownership for every step. Its receipt separates
edit-through-copied-result timing from untimed full-redraw verification and
explicitly leaves native presentation/internal transient memory unmeasured.
The first setup did not launch a case; the next accidentally included the generic
animation prelude and is invalid. Both fixture issues are corrected. The region/
fixture checks pass (11), using the bundled Python for the Pillow-dependent test;
the existing fixture extraction was corrected for its changed animation condition.

`local-region-edit-small-base-isolated-20260912` and
`local-region-edit-small-candidate-20260912` pass all four exact checks at 640x480,
width 128, over the complete 100x100 world. Inputs, sources and binaries match;
all runtime verification passes and no competing GPU/build process was observed.
The accepted shadow path is enabled in both arms. Their `comparison.json` is
one small matched pair, not repeated dense acceptance.

Distant edit/reversal latency is 45.636 / 46.296 ms control versus
24.434 / 24.973 ms candidate. Main-map drawing was already avoided in the control;
the eliminated work is rebuilding six unaffected animation backdrops. Candidate
backdrop hits/misses are 6/0 versus control 0/6. Visible edit/reversal is
1347.946 / 1359.989 ms versus 1317.981 / 1338.462 ms. Both build 80 geometry tiles;
geometry alone takes roughly 1.12–1.14 seconds. Those counts explain remaining
work, not a performance pass. That small comparison alone was provisional. The dense repeated edit and
unaffected-navigation results above now close this experiment at their stated
scope. Broader dependency/effect/lifecycle coverage remains to be earned.

### Resource shadow material variant — retained by user decision

`resource-shadow-material-build-20260912` is the final shadow-only isolated build.
`resource-shadow-material-boundary-20260912` passed all six independent full-
redraw comparisons with confirmed variant execution and unchanged inputs/sources/
binaries. Camera/guard movement, clock advance, removal and reappearance were exact.
The earlier resource checks passed (132 tests, one existing skip). The corrected
shader changes no source coverage, sample/depth order or rendering ownership.

`resource-shadow-material-reversal-candidate2-20260912/comparison.json` records
two profiling-off dense 14-request pairs, alternating arm order, fixed clock,
2240x1192, width 128, waves/reflections off and existing normal budgets. Controls
205.245 / 204.636 ms versus candidates 191.985 / 191.015 ms saved 13.260 / 13.621 ms.
All saved endpoint pixels/revisits were exact; complete inputs, sources and binaries
matched across the comparison; no competing GPU/build process was observed.
Sampled contiguous headroom was at least 1,690.660 MiB at host request boundaries;
internal transient/GPU-query coverage remains unmeasured with profiling off.

The analyzer classified this below its 20.494 ms usefulness threshold. The user
explicitly overrode that retention decision for this result. Keep the prepared
resource pass plus shadow variant under its existing scoped control and use it
as the candidate baseline in subsequent relevant experiments. This does not
retroactively accept failed body pixels, arrays, or unrelated historical probes,
or imply staging, install, launch or broader dynamic/effect acceptance.

The initial constant-material body/shadow candidate failed its first independent
full redraw: 77 pixels differed by up to two channel values.
`resource-material-boundary-first-20260912` preserves the mismatching images.
Restoring ordinary body shading isolated the issue: the shadow-only diagnostic
passed six checks. First shader-compiling runs changed runtime inputs and are
not stable-input acceptance. The body variant is removed, with no tolerance change.

### GPU region batch — rejected and removed

The candidate grouped up to four independently guarded texture-array layers for
body/shadow draws and reconstruction. It preserved each 136-pixel projection,
MSAA4 depth/sample contract and output crop, reserving 17,756,160 scratch bytes
inside the existing backdrop allowance. A revision removed a redundant cached-
background copy by restoring directly into the array slice. Neither version
established a useful whole-workload improvement.

`resource-gpu-batch-boundary-20260912` and the revised
`resource-gpu-batch-direct-boundary-20260912` passed six exact independent full-
redraw comparisons with actual batch execution and unchanged inputs/sources/
binaries. Resource checks passed (132 tests, one existing skip); focused region,
publication and budget checks passed (17). The first boundary run was exact but
changed shader cache inputs and is not acceptance evidence.

The initial four-column pair averaged 352.908 ms control / 381.435 ms candidate.
The deciding direct-copy 14-request repeats are in
`resource-gpu-batch-direct-reversal-candidate2-20260912/comparison.json`:
controls 196.547 / 194.388 ms, candidates 537.880 / 194.020 ms, paired savings
-341.333 / 0.368 ms. Saved images and revisits were exact, inputs/sources/binaries
verified unchanged, and no competing GPU/build process was observed. Sampled
contiguous headroom was at least 1,688.898 MiB at host request boundaries;
internal transient/GPU-query coverage was not measured with profiling off.

The first candidate sequence included a 3,754.416 ms readback wait; it did not
recur in the second sequence. Its cause is unestablished, so it is not discarded
as an environment fault. Even the unstalled repeat failed the usefulness gate.
Array code, adapter and test changes were removed; original shader outputs were
regenerated. The rejected source patch and new source files are preserved under
`resource-gpu-batch-direct-build-20260912/rejected-source/`. No production
optimization, navigation gate or live integration is claimed.

### Prepared resource pass — retained with the accepted shadow variant

The candidate now prepares ordered, region-specific body/shadow draw lists and
submits them through an explicit GPU pass using the existing shader/depth/guard
contract. Consecutive regions share exact source-shadow page demand in batches
within the existing 32-page atlas. A static backdrop miss invalidates the active
batch before animation resumes. The synchronous pass borrows pinned pose buffers;
its retained draw/page descriptions have a conservative 4 MiB allowance and fall
back to ordinary submission when exceeded. Wave frames retain ordinary submission
until their scene/depth pass contract is explicitly covered. No cache budget,
rendering ownership, staged binary or asset changes are involved.

`prepared-resource-batched-build-20260912` compiled successfully.
`prepared-resource-batched-boundary-20260912` passed all six exact comparisons
against independent ordinary-path full redraws, including camera/guard movement,
advancing clock, removal and reappearance; inputs/sources/binaries were unchanged.
`test resources` passed 132 tests with one existing skip. These establish the
initial correctness checkpoint, not dense navigation performance.

`prepared-resource-reversal-candidate2-20260912/comparison.json` closes the
preparation-only candidate: two profiling-off 14-request controls averaged
193.713 / 194.173 ms; candidates 198.170 / 193.230 ms. Paired savings were
-4.457 / 0.943 ms, below the 20 ms criterion. All four runs had exact saved
images/revisits, unchanged inputs/sources/binaries and no observed competing GPU
or compiler process. Host-boundary contiguous headroom was at least 1,691.730 MiB;
internal transient/GPU-query coverage remains unmeasured. The first four-column
candidate was exact but suffered a 3,841.913 ms readback wait (4,058.441 ms whole
request); the matched control was approximately 336 ms. That stall did not recur
in either reversal, but its cause is unestablished and it is not discarded as a
known environment fault. No useful performance pass or advancement gate is claimed.

The remaining dominant animation cost is GPU completion of regional rendering and
reconstruction, with per-request pose/selection work also present. Current preparation code
reduces setup calls but still executes the same serial region passes. The GPU-array
revision also failed, as recorded above. The preparation pass is now retained with the accepted shadow variant above,
under the same opt-in control. Its preparation-only organization is not separately
claimed as a speedup.
Focused region/publication/analysis checks passed 29 tests; the Pillow-dependent
check was rerun successfully using the bundled Python after system Python lacked
Pillow.

### Session tooling evidence (deliverable 2 complete)

Per-translation-unit include hashing and compiler/SDK stamps now reuse intact
objects. `session-incremental-seed-retry-20260912` compiled seven units in 37.345 s
wrapper time; `session-incremental-warm-20260912` compiled none in 3.108 s. Changing
only the preview compiled that unit alone. Recipe, dependency and object changes
reject reuse.

`session-four-verified-20260912/comparison.json` records four assets-loaded cases:
39.943 s wrapper total, 9.986 s per case versus 14.285 s for the matched fresh
one-shot. All initial/final images exactly match that independent fresh process;
full inputs, sources and binaries verified unchanged. This is about 30% less
iteration wait in this short comparison, not a renderer speedup or tail estimate.
Resets preserve budgets and clear mutable content/publication while retaining
assets/device. `session-prepared-verified-20260912/comparison.json` independently
matches the same fresh images after explicit warmup (5.056 s through its final
check), retaining 84.174 MB geometry and 30.628 MB ground content with no evictions.
Its single 146.326 ms small-view transition is not a navigation performance pass.

An earlier four-case run failed at teardown and entered Windows Error Reporting;
it remains invalid evidence. A subsequent watchdog exit-code reporting fault was
fixed and verified against success, nonzero exit and owned-process timeout cases.
The accepted runs above completed teardown under the bounded watchdog. The initial
transient teardown fault was not reproduced or localized; preserve that limitation
without reopening completed session validation as an unbounded investigation.

### Current evidence and limits

Ignored evidence under `Renderer/native/build/`:

- Bounded animation reconstruction is **rejected and removed**.
  `bounded-resource-boundary-20260912` passed six exact independent full-redraw
  checks; focused regressions passed (16), resource category tests passed (132,
  one existing skip), and navigation analysis/causal checks passed (16).
  Four-column time was 354.472 ms baseline versus 339.654 ms candidate, exact
  saved images but only 14.818 ms saved. Deciding profiling-off reversal repeats
  in `bounded-resource-reversal-candidate2-20260912/comparison.json` were
  195.805 / 198.330 ms baseline versus 189.115 / 190.932 ms candidate: only
  6.690 / 7.398 ms saved, below the 20 ms threshold. All four runs had identical
  saved endpoint pixels, exact revisits, unchanged inputs/sources/binaries and
  no observed GPU/build conflicts. Host-boundary contiguous headroom was at least
  1,689.332 MiB; internal transient/GPU-query coverage was unmeasured with profiling
  off. Reduced postprocessing area was not a sufficient architectural improvement.
  Advancing-clock expansion was not earned. The existing analyzer now supports
  explicitly requested profiling-off cases with constructor/host-only coverage
  labels and rejects accidental mixing with profiled cases.


- Static-pass sharing is **rejected and removed**. The preserved
  `shared-static-build-20260912` / `shared-static-boundary-20260912` passed six
  exact independent full-redraw comparisons (pixels, replacement flags and
  animation counts). Four-column paired repeats were 392.812 / 405.994 ms baseline
  versus 329.058 / 332.316 ms candidate, a 16.2–18.1% saving. However,
  `shared-static-reversal-candidate2-20260912/comparison.json` records the deciding
  14-request sequence: baseline 227.392 / 249.731 ms, candidate 244.933 / 246.393 ms.
  Paired savings were -17.541 / 3.338 ms, below the 23.856 ms usefulness threshold.
  All four sequence runs matched saved endpoint pixels, exact revisits, unchanged
  inputs/sources/binaries and zero observed GPU/build conflicts; minimum sampled
  contiguous headroom was 1,694.574 MiB. No budget was increased. Handoffs were
  often unused (114 of 180 initial captures); reduced duplicate submission did
  not yield a whole-sequence win. Advancing-clock dense expansion was not earned.
  The earlier failed baseline dispatch produced no child and remains invalid;
  its completed retry is `shared-static-four-base2-retry-20260912`.
- `retained-production-cost-20260912/production-cost.json`: one profiling-off
  14-request sequence against the cleaned renderer averages 193.617 ms. Phase
  means are geometry 34.906, submission 6.646, static readback 27.236 and animation
  composition 114.019 ms. Revisit geometry is 11.474–12.106 ms; revisit animation
  is 68.844–80.381 ms. All saved images match the profiled baseline and revisits
  are exact. This isolates production work; turning off diagnostic memory scans
  is not a renderer improvement. It is a single scoped cost check, not a repeated
  performance gate, independent full-redraw oracle or live-game measurement.
  Profiling-off omits per-frame address-space/GPU-query coverage; the separate
  profiled runs above preserve headroom evidence for the same budgets/workload.
- `retained-boundary-clean-20260912`: cleaned renderer build reuses the original
  DLL translation unit and rebuilds the retained preview fixture. Region/cache/
  causal tests pass (30); resource category tests previously passed (132, one
  existing skip). `retained-boundary-clean-check-20260912` passes all six exact
  reset/full-redraw comparisons with unchanged inputs/sources/binaries. The
  temporary handoff retirement test was removed with its
  rejected owner, preserving the original budget/LRU behavior test. No injected
  edit, staging, installation, game launch or reference replacement.

- `dense-route-current-20260912/results.json`: completed automatic batch, two
  repetitions in alternating arm order, 2240×1192, tile width 128, fixed clock,
  waves/reflections off, normal retained budgets. The 100×100 world supplies
  1,921 captured occurrences with 5 cities, 349 roads, 80 rails, 344 improvements
  and 150 resources. All 20 cases verified unchanged inputs/sources/binaries and
  no observed competing GPU/build process; minimum sampled contiguous headroom
  1,681.852 MiB. Repeated endpoints are exact; prepared-content endpoints match
  fresh-content rendering. Intermediate independent full-redraw parity is still
  unmeasured, and omitted-pixel arms cannot establish production correctness.

  | Arm | Four-column moves (ms) | Reversal mean per request (ms) | Decision |
  | --- | --- | --- | --- |
  | Full | 391.963 / 402.839 | 251.355 / 249.466 | Target unmet |
  | Route draws omitted | 394.056 / 396.714 | 243.751 / 250.359 | Reject primary target |
  | Route surfaces omitted | 405.555 / 406.747 | 245.846 / 250.916 | Reject primary target |
  | Prepared content | 345.472 / 309.586 | 224.613 / 227.262 | Useful four-column effect; reversal inconclusive at 10% threshold |
  | Half geometry pixels | 341.074 / 347.185 | 223.917 / 224.520 | Useful four-column diagnostic; reversal inconclusive at 10% threshold |

  Baseline reversal phase means: geometry 62.799 ms, submission 20.311 ms,
  readback 27.963 ms, animation composition 114.886 ms. Existing traces and source
  inspection show animated-region backdrop misses resubmit static geometry after
  main-map rendering. That observation led to the now-rejected sharing candidate above; the
  diagnostic batch itself established neither correctness nor speedup.


- `dense-route-quiet-20260912`: two uncontested, fully verified four-column cases
  completed on the 2240×1192 dense fixture, tile width 128, waves/reflections off,
  existing retained defaults and normal budgets. The 100×100 world supplied 1,921
  captured occurrences: 5 cities, 349 roads, 80 rails, 344 improvements and 150
  resources. Full rendering took 398.012 ms; route surface draws omitted took
  395.704 ms. Their pixels differ as intended. These are single samples, not a
  matched-repeat decision: route drawing showed no large effect in this pair.
  Full-render phases were geometry 114.525, submission 31.905, readback 71.509
  and animation composition 150.163 ms; sampled contiguous headroom exceeded
  1.8 GiB. Initial setup remains separately recorded. Runtime shader changes and
  another `native_preview` interrupted the third arm; it was rejected, and no
  performance conclusion is drawn from that arm or the incomplete matrix.
- `dense-route-diag2-20260912` contains the earlier overlapping batch, explicitly
  invalid for causal timing. The overlap watchdog was then added and correctly
  rejected `dense-route-serial-20260912` before rendering. It monitors known Lab
  renderer/compiler processes once per second, records conflicts and terminates
  only its own child. All diagnostic children are terminal. An initial startup
  exit in `dense-route-diagnostic-20260912` was not reproduced; the preview now
  reports the exact setup precondition on any recurrence.
- `dense-diag-setup-20260912`: isolated x86 build passed, with benchmark-only route
  surface and half-pixel controls. The category dispatcher exposes the fixed
  batch and the existing analyzer owns its comparisons. Focused causal/endpoint/
  build/damage tests passed (19); fixture/trace/worker checks passed (8).
  `test infrastructure` passed 123 tests with one existing skip. No injected edit,
  staged binary, reference replacement, installation or game launch.
- `endpoint-short-20260912`: 640×480, width 128, waves off, fixed-clock 14-offset
  reversal. All 14 requests and exact revisits passed, with zero fallback/recovery.
  Capture median 0.371 ms; request-to-checked-result median 164.303 ms, range
  31.628–858.733 ms. Initial render 5.550 s; before/after input verification 7.010 s;
  wrapper total 18.389 s. Revisit equality is repeatability, not full-redraw parity.
- `endpoint-four-column-20260912/analysis.json`: completed final endpoint layout,
  same small viewport and one four-column move. 759 captured occurrences, 728.304 ms
  request-to-checked-result: capture 0.447, worker rendering 725.935, queue 0.053,
  snapshot/drain 0.773, publication/preparation 0.905 ms. Within worker rendering,
  geometry 607.059, submission 14.240, readback 57.834, animation composition 39.297,
  unaccounted remainder 7.505 ms. The final delayed GPU sample is unmeasured; no
  blocking query was added. Native capture/presentation and independent pixel
  parity remain outside this tooling check. Full runtime input/binary verification
  passed; both builds produced the same initial image hash.
- `endpoint-accounting-complete-20260912/build-evidence.json`: isolated x86 `/O2`
  build passed. Compiler setup 2.110 s, DLL compile/link 22.620 s, preview 4.860 s;
  outer compile/dispatch 30.462 s. No staging, install, game launch or injected edit.
- Endpoint ordering/missing-query/invalid-span analysis, bounded trace behavior,
  dense fixture identity and delayed GPU telemetry tests pass (15 tests). Category
  dispatcher `test infrastructure`: 123 tests passed, one existing skip. The
  system Python 3.9 cannot run the telemetry test's newer tempfile option; the
  bundled workspace Python ran it successfully without changing that test.

**Dominant measured costs:** animation completion remains the largest phase in
ordinary dense navigation (about 114 ms in the earlier profiling-off trace).
Visible topology edits are dominated by geometry preparation: a representative
accepted-candidate trace reports 262 ms ground preparation, 773 ms natural-
geometry preparation and 100 ms upload for 81 rebuilt tiles. The river-field
hypothesis was subsequently rejected above; these phase counts do not select the
next task. Persistent appearance, compiled-content bindings and compact occurrence records are
retained. A compatible static batch pass is selected to consume those records and
reduce actual submission work; whole-request performance acceptance remains separate.
Local region validity now preserves unaffected backdrops on world edits; it does
not eliminate geometry rebuilding or establish fast native presentation.

The selected rendering migration remains retained route/object composition:
translate the valid static front, compose exposed strips/dirty bounds and keep
ambient/unit invalidation independent. Extend persistent content and local
dependencies where the diagnostics justify them. This is a step toward the
architecture, not its permanent definition. If evidence rejects the selected
causal target, replace this record's next task with one supported alternative;
do not start a competing sequence or the architecture's entire mechanism list.

| Capability / target | Evidence carried forward | Remaining limit |
| --- | --- | --- |
| Stationary retained front | Opt-in same-view ambient publication and independent cached-pose CPU composition are implemented; prior standalone minute-long 24-unit run reports 1.720 ms p95 render entry and 7.42 Hz fresh maps. | Still captures full native view; this is not live presentation or fast navigation. |
| Useful preparation | Existing `prefetch-guard2-replay-20260910/benchmark.log` records 176 prepared tiles before its first jump and exact return parity. | First three jumps were 707.534, 828.214 and 785.799 ms; short diagnostic, not a sustained gate. |
| Dense resident navigation | Retained overlap and bounded working sets already exist; preserve them. | Below-100 ms gate unmet; route/object composition and synchronous completion remain the selected causal target. |
| Regional geometry / zoom | Some structural sharing exists. | General regional batching shared across 128/160/192 is unfinished. |
| Native asynchronous presentation | Versioned publication and stale-ticket rejection exist in the DLL. | Native coordination, current-camera presentation and input-to-visible performance remain unverified. |
| Iteration tooling | Timing, incremental compilation, persistent sessions and the fixed dense causal batch are complete. | Additional tooling requires a specific missing correctness/measurement question for the selected rendering change. |

The preparation receipt is under `Renderer/native/build/`; older stationary
measurements are preserved in the linked experiment archive. These are scoped inherited
results, not a newly matched baseline/candidate comparison or current-binary
certification. No universal percentage-complete or overall speedup is claimed.

Preserve the staged DLL, rollback, licensed assets, source findings and existing
uncommitted renderer/build changes. Do not promote experimental options or alter
cache budgets, visual ownership or references during tooling work. Record future
progress here as capability, workload, evidence, unmet target and one next action.

## Preserved evidence

The [experiment archive](history/retained_experiments_20260910.md) retains the
original protocols, measurements and failed hypotheses. Read only the evidence
needed for the current mechanism; its embedded next-step instructions are historical.
