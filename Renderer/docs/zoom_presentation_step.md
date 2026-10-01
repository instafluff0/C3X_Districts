# Private zoom presentation admission

The existing frame-readiness contract improves the private retained-map preview.
The cheap path covers every observed guest refresh; **ordinary full refinement
still fails the 60 FPS gate**. This is an unpromoted HWND experiment. No game
binary, reference, asset, injected patch or scrolling implementation changed.

## Implementation and comparison

`tools/measure_zoom_presentation.py` freezes the preceding 219-file zoom-preview
prototype, adds the private admission/probe headers, and changes only the private
`resident_scene.cpp` presenter and measurement witness. Both comparison arms use
the same final DLL/client. The control retains its original two-buffer
flip-sequential HWND chain and `Present(1)`. The candidate adds the waitable flag,
sets maximum frame latency to one, and uses `Present(0)` after admission. Failure
to create or admit that chain is explicit; there is no ungated fallback.

`sandbox/zoom_presentation_permit.h` preserves a consumed auto-reset grant across
no-op opportunities. Its message-aware wait runs before input sampling and scene
submission, outside renderer/input locks. Message wakeups return to the caller
for dispatch. Only an actual `S_OK` Present consumes the grant or advances the
displayed zoom. Four executable host tests exercise ownership, saved grants,
message retry, timeout, unsupported/error handling and positive failure statuses.

The current fixture, GPU-ready seed, independent input clock, 60 ms debounce,
250 ms refresh-due policy, wide/close images and full-quality drawing are
unchanged. Day/night order reverses in two paired blocks for both frozen preview
and preview plus refinement. Sustained input, adaptive reversal and the existing
sixteen-actor night fixture follow. There are **22 quiet runs and 97,501 frame
records**, plus one separate actor-accounting run. Quiet captures/profilers stay
off after the common untimed seed. No assets were rebuilt.

Four QPC phases now measure setup, output draw, capture and DXGI Present
separately. Previously the first phase combined setup and output drawing and the
second was trivial unit-option handling. Admission WAIT is recorded separately
and remains included in successive return intervals. Existing six refinement
phases and snapshot submission are recorded without a completion probe.

## Delivery observations and timing

Every final Present returns `S_OK`; counters/statistics also return `S_OK`.
Full DXGI frame statistics, sampling QPC and submitted image IDs accompany source
and input metadata. Only advancing in-window guest refresh observations with
matching sync/presentation refresh timestamps enter the table. Latest statistics
are not an event stream: intervening IDs can be unobserved or discarded. These
are **observed delivery intervals**, not a count of all displayed images or host
physical scanout. The control can undersample intervening deliveries.

| Pooled observed intervals | Mean | p50 | p95 | p99 | Worst |
|---|---:|---:|---:|---:|---:|
| Noon frozen control | 25.82 ms | 18.44 | 47.88 | 50.52 | 99.99 |
| Noon frozen waitable | 16.67 ms | 16.66 | 18.25 | 18.53 | 18.62 |
| Night frozen control | 27.22 ms | 31.47 | 35.40 | 64.65 | 132.96 |
| Night frozen waitable | 16.67 ms | 16.67 | 18.30 | 18.58 | 18.66 |
| Noon refinement control | 31.01 ms | 31.41 | 51.10 | 133.40 | 166.94 |
| Noon refinement waitable | 18.89 ms | 16.67 | 18.51 | 116.69 | 133.34 |
| Night refinement control | 32.72 ms | 32.70 | 66.67 | 151.10 | 168.33 |
| Night refinement waitable | 19.15 ms | 16.67 | 18.59 | 116.67 | 149.94 |

Exact 16.667 ms exceedances are 124/264 and 128/264 for cheap candidate noon/night,
including timestamp jitter; neither has an interval above 25 ms or a skipped
observed refresh. Refinement has 123/232 and 119/228 exact exceedances, 8/9 above
25 ms, 5/6 above 33.333 ms and four above 100 ms in each hour. Refresh/ID steps
prove at least 20 noon and 23 night refresh opportunities without a new image;
31/34 refresh opportunities lack an intermediate observation. Those two counts
must not be conflated.

Waitable API returns average only 0.23 ms for the cheap path and 0.30–0.31 ms with
refinement. **Those thousands of returns are not displayed FPS.** `Present(0)`
readiness permits submissions more frequently than refresh on this VM. The
benchmark continues submitting held images; production must retain its existing
independent cadence/no-op suppression. Submitted-minus-observed ID spans do not
measure queue length. The bounded 100 ms observation tail does not close the last
submitted ID in any run, so final submission delivery remains unknown.

The control's cheap cost is mainly inside output drawing, around 19 ms average,
while DXGI Present is small. The candidate's cheap output is about 0.05 ms, but
refinement moves long backpressure into admission: ordinary maxima 147–159 ms,
busy 164.21 ms. It does not eliminate those pauses. EVENT queries gate completed
sources; their pending/ready observations are not GPU pass timings or scanout.

Ordinary candidate CPU draw/snapshot submissions average 11.86/11.80 ms noon/night
versus 26.09/25.86 ms control; candidate maxima reach 29.42 ms across the quiet
workloads. First observed changed images arrive 28–32 ms after ordinary input,
versus approximately 98–100 ms control (observation bounds). Ordinary current
quality returns after 74–78 ms and is first observed after 90–107 ms. Source age
still reaches 538 ms ordinary and 540 ms busy; whole-map animation remains held.

Sustained candidate observations average 16.67 ms, worst 20.68 ms, with every
observed refresh covered, fourteen redraws and final observed quality at 80.02 ms.
Adaptive reversal averages 19.32 ms, worst 149.14 ms, with final observed quality
at 99.66 ms. Busy averages 19.62 ms, worst 150.01 ms, and final observed quality
at 107.42 ms versus 182.75 ms control. Its ten redraws preserve sixteen moving
actors/85 parts plus the four original synthetic actors. Separate accounting
retains 212 main, 106 reflected and 170 shadow unit draws per redraw, with
79,310/39,655/62,942 triangles. No actor, effect or quality policy was dropped.

## Next bounded rendering split and production scope

The smallest proposed split is an ordered pass/layer/record cursor in
`sandbox/fresh_pipeline.h` (`issue_records` and vegetation batch flush). Bound
records examined as well as records issued, retain nested instance position,
keep paired vegetation depth/color draws together, and clear each target once.
Freeze frame/projection/lighting/animation time and geometry leases for the job;
rebind state after preview drawing and publish coverage only at full completion.
Reflection/main material, scene, water, units and reconstruction order must stay
intact. Bound GPU batches too: moving CPU work alone leaves the admission stall.

For that minimal cursor proposal, still-unsliced preparation—including visibility,
resource poses, city-light selection and shadow preparation—reaches **3.153 ms**
in the quiet candidate runs. Snapshot submission reaches 1.361 ms and reconstruction
0.062 ms. These are measured whole-phase CPU upper bounds, not proven indivisible
API costs. The longest current record-containing phase is reflection at 11.268 ms;
the whole current submission reaches 29.420 ms. Full-screen relight/resolve/copy
and reconstruction remain indivisible GPU operations whose isolated duration is
unmeasured. No resumable implementation, timer sweep or worker sweep was started.

The game-facing composition presenter already has the waitable/max-one-latency
contract in `native/c3x_renderer.cpp`; copying this HWND experiment into game code
would not add that capability. This slice's concrete changes concern the private
presenter and witness. A later game candidate must admit retained-preview work
before expensive refinement, preserve successful-Present picking/native UI,
and integrate bounded refinement at the existing composition owner. Fresh dynamic
layers and native composition remain separate, unqualified work.

Evidence is in `Renderer/.cache/zoom-presentation-step/`: frozen sources/binaries,
per-run logs/receipts, two combined CSVs, `final-summary.json`, phase and delivery
timelines. A preliminary 600-record run ended before input and is excluded; two
preliminary VM dispatch failures left no client running. Their sources, binaries
and receipts remain preserved. All 29 admission, policy, WARP coverage/depth,
projection, picking, composition and ownership tests pass in 70.386 seconds.
Preservation verifies 47,224 accepted inputs (12,238,666,273 bytes), 193 runtime
files, 219 prior and 221 current sources, all three earlier evidence manifests and
the staged tuple. No injected compilation was needed. User patch-table action: none.
