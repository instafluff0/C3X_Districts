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

## Current GPU composition investigation

The user authorized the bounded [GPU composition probe](gpu_composition_probe.md)
and necessary verified patch-table additions. The standalone path passes exact
RGB composition and native JGL fill/copy checks. HDC-only replacement cannot cover
native pixel-writing operations. GPU destination paths eliminate application map
readback in the fixture, but variable completion stalls prevent a reliable speedup
estimate. Typed pass-through JGL access/copy/fill/sprite
hooks and a verified GOG final-transfer hook feed a bounded caller-thread capture that
records map/screen identities, attempted copy dependencies and unattributed CPU
access, flushing diagnostics after native transfer. It preserves existing pixels
and UI timing ownership; it does not yet remove readback or establish live coverage.
The intended next boundary is hybrid composition: retain CPU-generated static UI
textures and translate the operations that depend on the animated map, preserving
native demand and complete UI transactions. This is not a wholesale UI rewrite.
Observation candidate `038faec0…0be2e2` passes native-hook execution and exact
production-scene parity and is staged with explicit user permission. Installation
and live capture wait for the user's return. Meanwhile the reusable packed GPU
executor replaces the isolated prototype's earlier GPU arm: bounded resident
images, source revisions and exact copy/fill/key/invert/save/restore transactions.
Native 16-bit and full-frame RGB oracle checks validate admitted operations.
The actual injected hooks now execute admitted copies, fills and ordinary keyed
image draws through `native_image_adapter.h` in the isolated JGL harness. Fresh
lifetimes own GPU destinations; escaped CPU pixels/HDCs force a synchronized,
permanent return to CPU ownership. Complete word comparisons catch retained-pointer
source edits and skip unchanged uploads. Clip metadata has its own direct hook.
The existing GPU worker now accepts GPU-only map demand and copied composition
packets. It unwraps the finished retained surface on the GPU, imports an immutable
map into the shared packed-image executor and publishes a ticket plus native
ownership metadata. UI images survive subsequent GPU frames; stale tickets and map
writes reject. CPU fallback invalidates its stale bitmap before rendering again.
The live loader remains unbound: connect native adapter packets, complete surface/
palette/access coverage, device-loss recovery and the native final GPU transfer.
Source comparison cost is still on the caller. This is demonstrated reuse and
removed map readback in the integrated renderer fixture, not a game speedup.

## Actual code responsibilities

Completion means ownership and behavior for admitted categories, not permanent
GPU residency or knowledge of uncaptured gameplay.

| Responsibility | Implemented ownership | Remaining limits |
| --- | --- | --- |
| Persistent world/instances | `CapturedScene` retains canonical appearance/revisions and borrowed compiled handles. `ResidentContent` owns complete reusable ground, city, route, improvement and natural content independently of observations and camera departure. | Complete tile reuse covers the full-detail 2:1 city profile at native zooms (128/160/192; admitted class ≥96). Other bases retain existing paths. Captured surroundings only; units retain their separate playback/pose owner. |
| Local validity | Appearance, semantic neighbors, city exclusions, coast/world samples, river-cell proofs and exact scaled native-anchor dependencies guard complete compiled hits. Local changes retire known-invalid alternate views; fresh capture validates every publication. | Asset/device/world-basis changes remain global. Unobserved appearance cannot authorize output. |
| World → view selection | Immutable content owns persistent bounded world-cell membership. Current authoritative occurrences supply eligibility; selected inputs feed existing ordered passes. Animated body/shadow/filter bounds are selected independently of surrounding static coverage. | Non-affine and unindexed inputs use the existing view index. Dynamic selection uses small scans. No complete native active-unit pass yet. |
| Compatible submission | Shared meshes, tree instancing, ordered materials, common depth and bounded shadow-page batches serve foreground and background through the same path. | Individual demanded unit misses and optional reflection profiles retain their existing execution. |
| GPU/output reuse | Stable surrounding static color/depth, selected animation, sparse restore, pending finish damage and consolidated readback. Selected updates publish only their fresh viewport; untouched donor margins keep their true old clock. | GPU allocation and full hardware MSAA resolve still cover the wide working surface. Physical targets are not narrowed. Tested retained profile has waves/reflections off. |
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
