# Retained renderer: current implementation and priorities

## Objective and boundaries

Complete world → view → pass → publication with bounded ownership, full detail
and source animation timing. Measure work eliminated, work prepared before demand
and caller delivery separately. Civ III supplies demand; workers never advance
gameplay, notify it or request redraws. This connected implementation objective
supersedes the former single-experiment queue.

Preserve AGENTS.md, native overlays/visibility/picking, generic assets, accepted
visual contracts and deferred wonders/Districts. Tested DLL staging is authorized;
installation and launch remain the user's actions through ordinary `INSTALL.bat`,
without environment settings. The authorized Advisor lifecycle hook reuses its
existing patch-table addresses; see the patch ledger.

## Current GPU integration

**Accepted correctness checkpoint:** the user confirmed that preserving JGL's
existing graphics owner fixed the selective UI corruption. The misnamed factory
and 8-bit-default regression are preserved in `ui-owner-fix-checkpoint` with the
before/after TCC test. Temporary sprite scans and GPU oracle readbacks have now
been removed from game execution; standalone correctness oracles remain.

**Step 1 — live GPU lifetime integration:** direct native load/unload hooks now
attach tracking after the original graphics factory returns and before native
canvases are created. Palette metadata no longer falsely revokes eligibility;
public CPU escapes still do. Same-size native init cannot erase an escape, and
DLL reload clears prior lifetime evidence. No callback requests a Civ III draw.
The connected fixture now enters through this production bootstrap rather than
manually providing lifetime registration. Tests include original owner continuity,
native config-off pixels, load failure/reload, actual storage replacement and
zero-readback admitted composition. See the [patch ledger](civ3_patch_dependency_ledger.md).

Actual gameplay confirmation remains pending: the last supplied pre-fix log used
CPU composition and GPU snapshot presentation. New `composite gpu_map=1` and
`native-resident-present` records distinguish the complete resident path from
compatibility presentation. Do not equate passing connected tests with complete
coverage of every native screen. The selected unit shadow pass below advances
compatible submissions at full detail; map demand now dominates scrolling.

The earlier pre-residency debug capture and accepted UI repair are preserved
in the startup and UI-owner checkpoints. They are not controlled comparisons of
the current GPU path.

The [GPU composition implementation](gpu_composition_probe.md) now has a production
native owner. `composite_custom_renderer_frame` offers its authoritative capture
through resident-map prepare, validates the returned replacement flags, then
commits the GPU map without acquiring a CPU destination DC. Cancelled validation
does not insert pixels. Unsupported admission retains the existing bitmap path;
a GPU failure does not authorize stale native pixels.

The implemented owner admits the map and its copy/save family on lifetime evidence;
the supplied pre-fix capture lacked that evidence; corrected startup now records it.
Unused canvases allocate no GPU images; unrelated UI fills remain CPU-generated
sources. External pointer/DC access revokes eligibility until storage replacement. One caller
owner routes native copies, sprites, cached text, prepared units and final Graphsy transfer to
the existing GPU worker, with paired native-word/full-color images. Ordinary indexed sprite
indices 254/255 are transparent independently of palette RGB. Native stretching,
full-color keyed copies and palette/whole-image fills use this same submission
owner. Transparent self-draw keeps native traversal through an explicit barrier.
Map colors retain
the original BGRA precision. Positive indexed scaling follows JGL's source programs;
mirrored sources and unsafe trimmed edges retain explicit fallback. The separate
HUD alpha programs now blend resident underlays/destinations on the GPU, preserving
native arithmetic and full-color map contribution. The staged candidate also
routes translucent map-label panels and border/indicator lines through this owner;
axis-aligned lines reuse fills and diagonal coverage reuses the sprite source.
The staged candidate also handles destination-dependent lookup panels, indexed
effects and ordinary/scaled native FLC palette/shadow drawing through one table
program, retaining exact native words and full-color map contributions. Connected production checks
verify these forms without execution readbacks. Single-key UI artwork, solid
sprite masks and native map shadows now use this owner too. CPU-owned picking
masks remain native; full FLC and small shadow lookup assets coexist.
No renderer callback requests a draw; transfer follows final tooltip/cursor drawing. CPU UI
sources share the GPU presenter, including partial updates over resident maps.
The final-presentation hook binds the process-owned GPU presenter on configured
UI demand, including before the first map and after scene unload. It needs no
map shaders; config-off menus remain native. Foreground transfer preserves
interrupted camera requests instead of selecting GDI because preparation is busy.

Reset, configuration and detach drain this owner before retiring its GPU session.
Partial GPU-to-GDI handoff preserves the last displayed image outside the native
update; only that explicit fallback snapshots the display. Failed barriers deny
native leases and defer reinitialization/destruction; device-loss reconstruction
is deferred by the user; existing failure guards remain. The existing 64 MiB composition budget and bounded startup
registry remain in force.

The production exports pass connected JGL tests at 640×480 and 1440×900: cancelled
publication, exact metadata, unchanged CPU map storage, native copies/units,
second-view reuse, cached native labels, displayed pixels and reset handoff. Native dispatch tests check
the actual injected branch and its epoch/clock/fallback behavior. This demonstrates
normal-path work removal in the harness; the paired comparison below measures the
complete rendering request, not live-game speed.

Native labels retain Windows font/layout/smoothing and use resident response data
for background-dependent blending. The cache holds at most 32 runs / 8 MiB within
the existing GPU budget; a miss uses under 2 MiB CPU scratch. Native/map pixels are
never read for glyph preparation. The user accepts slight text-edge color changes.
Rotated/transformed text, complex clips, current-position/RTL drawing and oversized
runs retain explicit native fallback; ordinary static UI can still be CPU-generated.

Unit bodies and shadows finish and remain on the GPU. Existing CPU helpers
compile compact selected caster records alongside body geometry. One instanced
shadow pass preserves full resolution and the CPU barycentric coverage rule;
material and finishing passes consume its resident height texture. CPU callers
retain the original raster/finishing route. Pose content distinguishes CPU/GPU
representation, while the bounded 64 MiB / 512 completed-texture owner and native
source timing/anchors remain unchanged.

Prepared-area publication now shares one content/spatial validity owner for CPU
and immutable GPU storage. GPU selection retains a bounded texture view; the worker
imports only its selected rectangle into the native composition session. Caller UI
submissions preserve queued preparation, while incompatible map demand cancels it.
Cold demand and preparation use the same working extent; first demand establishes
reusable coverage. A clock-aware native export returns the actual adopted ambient
sample. Connected 640×480 and 1440×900 tests verify asynchronous refresh adoption,
exact native CPU-reference pixels and fallback; the injected smoke also passes.
The previous unit/text checkpoint remains the reproducible control.

The step-1 control `0bd678b8…841e4f`, exact sources, original paired benchmarks,
and injected smoke remain in `native-startup-integration-checkpoint/`.
Startup/config-off/reload, native GDI handoff and injected compilation passed;
this step changes no injected source or patch-table entry. New `unit-body`
counters identify GPU shadow passes, compact input bytes and separate CPU-route
height uploads. Counters establish work placement, not elapsed speed.

Speculative unit GPU submission now consumes only ready CPU content. Unready
predictions stay in the finite queue; helper completion wakes the internal GPU
worker under its queue lock. No timer or native callback is added. CPU input
leases end before completion is observable; callback removal joins notification.
The ready-content correction and its measurements remain in
`ready-content-checkpoint/`; the new shadow pass uses that same scheduling owner.
The unrelated terrain-boundary mismatch remains recorded.

Opacity-driven command-panel/UI transitions (`FUN_005f8a70`, JGL sprite slot 37)
now feed the existing source owner and GPU blend submission. Native sixteenth-step
weights, endpoint quirks, flags, indexed transparency, clipping and positive scaling
match the pinned JGL oracle. Full-color map contributions remain resident.

**Remaining integration:** validate the complete path during ordinary game UI,
scrolling and picking. The native/connected fixtures establish exact tested behavior,
not coverage of every game screen. Unsupported text/sprite states and explicit
public pixel/DC escapes retain CPU barriers; static CPU source preparation is
intentional. Do not turn native source-only preparation or unsupported JGL stubs
into another primitive-porting queue.
Device-loss reconstruction is user-deferred. Map rendering now dominates the
scrolling workload; general nonblocking native scrolling and scene-wide unit
collection remain incomplete. Earlier controls stay in `whole-frame-control/`.

Earlier opacity and lifecycle controls remain in
`native-opacity-completion-checkpoint/` and `native-ui-lifecycle-checkpoint/`;
the accepted corruption fix remains in `ui-owner-fix-checkpoint/`. Original
observer controls, rejected approaches and expensive source findings survive.

## Step 2 — selected unit shadow pass

The bounded Warrior CPU probe attributed 20.42 ms/pose to the 1536² shadow
raster, versus 0.09 ms sampling and 0.03 ms vertex preparation on the Mac.
The implemented correction moves raster ownership to an explicit GPU pass.
Helpers retain compact immutable caster records; one instanced submission feeds
the existing material and resident finishing passes. Full resolution, native
anchors, source timing and CPU fallback remain. The Warrior idle input replaces
a 9 MiB height upload with at most 98.25 KiB of caster records. Record preparation
and GPU storage are each bounded to 16 MiB; the existing pose budgets remain.

The focused oracle checks exact coverage and 221,488 finished pixels across
128/1536 shadow resolution, direction and zoom. Maximum height rounding error is
1.19e-7. A direct final-edge comparison preserves CPU coverage despite the VM
reassociating `1-u-v`; no precision emulation, tolerance expansion or timing guard
is needed. Category checks pass (130 tests, one existing skip); connected native
pixel checks pass. Candidate `75ecd309…c66bed` is staged for ordinary `INSTALL.bat`.
The exact sources, DLL, tests, control and measurements are preserved in
`native/build/gpu-composition/unit-shadow-pass-checkpoint/`.

The 1120×1192 harness compares full requests with eight animated units and native
UI/final transfer; capture/source setup are outside timing. Each build runs
CPU/GPU/GPU/CPU, with 64 measured requests per route/workload. Below compares the
previous **GPU route** with the final GPU route, not CPU versus GPU:

| Workload | Mean request old → new (ms) | Request p95 old → new (ms) | Mean desktop completion old → new (ms) |
| --- | --- | --- | --- |
| Stationary animation | 38.14 → 8.49 | 134.65 → 14.40 | 47.34 → 17.64 |
| Dense scrolling | 61.98 → 49.23 | 94.92 → 65.08 | 73.08 → 59.11 |
| Local-change sequence | 38.80 → 12.65 | 136.44 → 52.11 | 46.82 → 21.28 |

A preceding candidate run measured means 8.59/50.19/11.25 ms; local-change tails
vary with the actual edit. Final idle unit delivery averages 0.83 ms versus
29.03 ms in the control; scrolling map delivery still averages 43.92 ms. Both
routes retain matching ambient sample ages. The admitted path has zero execution
readbacks and unchanged native CPU map/screen storage; sampled contiguous address
space stays above 1.43 GiB. This is harness latency, not gameplay FPS or physical
scanout; actual gameplay remains pending. Parallels GPU timestamps remain
unreliable; these timings use whole-request QPC and desktop completion.

This completes the unit shadow pass, not scene-wide unit collection or shared
resident vertex submission. Those remain distinct architectural responsibilities;
vertex caching alone would not have removed the measured dominant CPU cost.

## Actual code responsibilities

Completion means ownership and behavior for admitted categories, not permanent
GPU residency or knowledge of uncaptured gameplay.

| Responsibility | Implemented ownership | Remaining limits |
| --- | --- | --- |
| Persistent world/instances | `CapturedScene` retains canonical appearance/revisions and borrowed compiled handles. `ResidentContent` owns complete reusable ground, city, route, improvement and natural content independently of observations and camera departure. | Complete tile reuse covers the full-detail 2:1 city profile at native zooms (128/160/192; admitted class ≥96). Other bases retain existing paths. Captured surroundings only; units retain their separate playback/pose owner. |
| Local validity | Appearance, semantic neighbors, city exclusions, coast/world samples, river-cell proofs and exact scaled native-anchor dependencies guard complete compiled hits. Local changes retire known-invalid alternate views; fresh capture validates every publication. | Asset/device/world-basis changes remain global. Unobserved appearance cannot authorize output. |
| World → view selection | Immutable content owns persistent bounded world-cell membership. Current authoritative occurrences supply eligibility; selected inputs feed existing ordered passes. Animated body/shadow/filter bounds are selected independently of surrounding static coverage. | Non-affine and unindexed inputs use the existing view index. Dynamic selection uses small scans. No complete native active-unit pass yet. |
| Compatible submission | Shared meshes, tree instancing, ordered materials, common depth and bounded shadow-page batches serve foreground/background. Unit caster inputs now use one compatible GPU submission per pose. | Body materials still submit per part. Scene-wide unit collection and optional reflection profiles retain their existing execution. |
| GPU/output reuse | Stable surrounding color/depth, selected animation and pending finish damage feed immutable GPU publication. Resident map/unit output, native UI composition and final transfer share the GPU path; CPU callers retain consolidated readback. Untouched donor margins keep their true old clock. | GPU allocation and full hardware MSAA resolve still cover the wide working surface. Physical targets are not narrowed. Tested retained profile has waves/reflections off. |
| Preparation/native delivery | Bounded CPU helpers and one GPU owner prepare copied nearby/zoom views and unit poses. Fresh native requests pull compatible publications. Current refresh can interrupt unfinished prospective work at safe points. | Cold/outside-coverage views still use exact blocking fallback. Submitted GPU work is not preemptible; preparation can delay ambient freshness. Native delivery is implemented for prepared coverage, not general nonblocking scrolling. |

Native selected idle/work loops preserve source frames/duration; inactive unselected
units freeze. Directed actions keep native cursors/anchors. The authorized GOG
adapter offers unit visuals at 33 ms, preserves native advancement at 66 ms and
map sampling at 15 Hz. Extra work suspends during guarded native UI operations.
No copied animator or competing presenter. See the
[patch ledger](civ3_patch_dependency_ledger.md) for capabilities and fallbacks.

## Completed world/view and zoom extension

Optional `c3x_renderer_prepare_view` copies caller-supplied alternate captures.
The native adapter supplies the existing 128/160/192 zoom levels using native
fixed-point anchors and full captured appearance. Query/admission means queued or
retained, never permission to display. Actual capture must prove exact projection,
visibility, content, ownership and lifecycle compatibility before acquisition.

Three bounded prepared views retain useful projection results and switch through
the same owner. Geographic padding is still at most 128 pixels per axis within
2240×1192; this step did not enlarge the surrounding radius. Combined committed
CPU areas are capped at 64 MiB, each at 32 MiB; one additional in-flight area is
capped at 32 MiB. Two queued prospective snapshots share a 16 MiB payload limit;
one latest current-refresh snapshot is separately bounded by API maxima (under
8 MiB). Existing GPU assets/targets, front publication and transient copies are
additional. Eviction invalidates borrowed handles without destroying world identity.

Selected refreshes preserve static output outside damage. Pending off-view damage
is bounded and finished when selected; it cannot resurrect an old pose. The wider
donor is refreshed periodically, with age used for scheduling rather than static
invalidation. Current requests take priority over unfinished future zoom builds.
The complete world-content owner now also removes standard-zoom recompilation;
these bounded prepared views continue to own distinct finished pixels.

## Native UI checkpoint remains independent

The prior game log contained 2,135 map passes (12.05 ms median, 173.60 ms p95,
4.59 s maximum), 900 cancelled area jobs and over 54,000 detail lines. Native
Advisor, outer-popup and command-button function-lifetime guards now suspend extra
renderer work during UI construction; default tracing samples detail. No timeout,
GDI flush or native painting replacement was added. These hooks are unchanged here.
Actual Advisor/popup/button painting and the reported black bars/shadow flicker
remain the user's native checkpoint; passive `map-black-span` diagnostics remain.

The prior UI build established no performance gain: scrolling mean/p95
13.30/16.81 → 13.31/16.29 ms; eight-body idle 5.62/7.07 → 5.50/7.07 ms.
A one-second speculative retry delay was rejected (13.30 ms either way).
Control `d3acf940`/`cec59b1b…820dea`, checkpoint DLL `ccc5c1dd…d2817b`, full
hashes, comparisons and staging receipt remain in `native/build/native-regressions/`.
Its nine prepared camera images matched control, and unit/GDI/injection checks
passed. Independent prepared-area differences predate this step; no live UI or
speedup acceptance is inferred. No game was launched.

Prior world/view/zoom evidence remains in `native/build/world-view-zoom/`:
`comparison-v12.json`, `selection-diagnostic-v12.json` and final `v12` runs preserve
full measurements and source/binary identities. Prepared first closer zooms went
from 1.79/2.30 s to 6–7 ms after a separate five-second preparation opportunity;
nearby scrolling/idle did not improve and cold construction remained expensive.
The 20-call diagnostic retained 19 map jobs, zero geometry builds/uploads/static
submissions, and 122.35 million resolved pixels in both; reduced CPU donor copying
was not a broad GPU speedup. GPU timestamps remain unreliable.

## Complete representative world → view → pass path

The existing resident owner now stores full tile content. Ground uses a canonical
128-pixel basis; natural/city meshes use world coordinates; legacy features retain
their separate height/depth rule. Shader occurrences apply the current native
anchor, zoom and viewport depth basis. Unchanged standard zooms bypass all tile
compilation and upload. Full detail, source assets and pass ordering are preserved.
The redundant projected CPU ground-grid cache is bypassed for this path.

`WorldPassIndex` retains immutable content membership in normalized isometric
cells. Eviction removes membership and invalidates borrowed handles. A fresh view
proves an affine native-anchor basis and supplies observed eligible occurrences;
queries merge these with residual inputs before compatible submission. Its 16 MiB
membership cap and bounded view map fall back to the existing bounds selection.
No retained world record grants visibility or replacement ownership.

CPU terrain preparation uses the same camera-independent input basis. Completed
work remains useful across nearby views/zooms and interrupted prospective jobs.
This does not move all object compilation to helpers: first-time city/route/object
construction still runs on the GPU owner, and residency remains bounded.

A bounded diagnostic exposed a new interaction: faster alternate-zoom compilation
let speculative GPU jobs replace the single current working surface during camera
motion. Prepared scrolling rose from 13.43 to 15.99 ms mean and sampled map age
from 596 to 2,047 ms; the instrumented repeat reproduced it. Current camera/content
changes now supersede unstarted other-zoom pixel jobs; completed world content
survives. Fresh offers resume after a stable view, and already prepared zooms remain
available. This is request-based admission, without a new timer or native hook.
The rejected intermediate is preserved in `candidate-v9` and the paired
`*-scroll-diagnostic` evidence directories.

Control: source `f511a547`, DLL `ccc5c1dd…d2817b`, preserved with its original
harness in `native/build/world-content/control/`. Matched comparisons use that DLL
and the candidate with the same updated harness/runtime shader inputs; old bindings
leave the new projection branches inactive. No output-helper optimization or new
native patch is part of this step. Results and the final staging receipt are kept
in `native/build/world-content/`; the existing native UI checkpoint remains pending.
General nonblocking cold/outside-coverage scrolling and complete active-unit demand
collection remain separate integration responsibilities.

## Final measured checkpoint

Final DLL `55276aec…f196ca`; control `ccc5c1dd…d2817b`. Whole-request timings
use 1440×900 except the 640×480 local-edit fixture. Setup, warmup and independent
redraw oracles are excluded. `native/build/world-content/final-comparison.json`
and `final-invocations-v12.json` preserve exact samples, inputs and binary hashes.

| Workload | Control → final | Interpretation |
| --- | --- | --- |
| First closer zooms, 192 / 160 | 1,282 / 1,697 → 193 / 294 ms | 6.7× / 5.8× faster; 363 / 479 tile builds become zero. |
| Dense scrolling, 14 calls, mean / p95 | 169.29 / 469.63 → 171.03 / 469.01 ms | No gain; newly exposed content still requires compilation. |
| Local city/forest edits, six render calls, mean | 43.06 → 44.20 ms | No gain; two affected tiles rebuilt, 261 reused. |
| Prepared scrolling, 48 calls, mean / p95 | 14.16 / 18.99 → 16.76 / 20.41 ms | Caller latency regresses; maximum sampled map age improves 606 → 311 ms. |
| Eight-body idle, 60 calls, mean / p95 | 5.53 / 7.03 → 5.42 / 7.16 ms | No established gain. |

The correction removes the two-second speculative freshness stall, with the
remaining scrolling latency/freshness tradeoff explicit. Cold initial construction
and warm zoom GPU completion remain expensive; a broader live speedup is not
established. CPU helpers are disabled for zoom/dense/edit work-elimination cases;
prepared scrolling/idle use production defaults. No reduced detail or new targets.
Across the three standard zooms, geometry residency is 388 → 234 MB; persistent
membership peaks at 5.51 MB in the measured diagnostic cases. The final cases retain
at least 1.39 GB of contiguous 32-bit address space, above the 512 MiB floor.

Windows build, x86 worker/index execution, 30 focused local contracts, night/wrap/
small-zoom and 288-pose unit checks pass. All three standard zooms and ten local
edits match independent redraws exactly. The smaller-zoom fallback has one total
channel-level error within its existing tolerance. New control-image precision
differences are recorded: 1–23 pixels, at most three channel levels; prepared views
have 0–2 changed pixels, at most two levels. The intermediate cliff depth-basis
error was corrected. Existing prepared-area versus cold-render differences remain;
references and prior visual acceptances are unchanged. The evaluation comparison
is `native/build/world-content/visual-comparison.png`; visual acceptance of these
new precision differences remains the user's decision.

## Preserved controls and closed findings

[Checkpoint history](history/retained_renderer_checkpoints_20260914.md) preserves
prior measurements, accepted visual differences and rejected intermediate designs.
[Completed output work](history/retained_output_completed_20260913.md) records the
119.40 → 103.28 ms dense gain and no stationary gain. Do not reopen closed output
or resolve experiments without a materially different mechanism. Global 30 Hz map
sampling regressed warm requests 3.97 → 7.46 ms and remains rejected. Parallels GPU
timestamps/event-query attribution is unreliable; whole-request CPU wall time is
the performance authority, and readback includes queued GPU work.
