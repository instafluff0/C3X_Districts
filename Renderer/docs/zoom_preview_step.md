# Retained-image zoom scheduling gate

Implemented and measured a private prototype from the accepted renderer baseline
(called C7 in the earlier review). The gain is substantial, but the **60 FPS gate
fails**. No binaries were promoted or installed. Further close-zoom expansion,
fixed-zoom scrolling reuse and underlay tuning were not started.

## Implementation and scheduling

`sandbox/zoom_preview_gpu.h` retains at most two original, fully rendered BGRA
maps and guarded D24 snapshots. Filtering always samples an original completed
image. Requested/completed relative scale is independent of the absolute 1–3
projection range, including relative scale below one. Uncovered samples are
transparent, never edge-stretched. A coherent 1× source remains available while
the other slot is refined; allocation padding is not treated as rendered coverage.

`sandbox/zoom_preview_state.h` separates requested, presented and completed
projection/identity. Obsolete requests coalesce. Overwritten sources are removed
before reuse; only successful presentation changes the displayed transform. Scene,
visibility, viewer, camera, target size and native tile size participate in identity.

The private scheduling witness starts refinement after 60 ms without input. A
250 ms content age makes a refresh due, including sustained input; this is **not a
hard age guarantee**. One genuine GPU job remains in flight. A nonblocking D3D11
EVENT after full rendering and snapshot copy gates completion/publication. An
adopted result receives a display opportunity before another refresh can overwrite
it. No worker thread is used to imply independent GPU rendering.

Input comes from an independent QPC/Sleep producer. Full draws sample animation
from actual elapsed wall time, not frame count. Producer event IDs remain distinct
from coalesced policy IDs. Quality is checked against input actually received at
Present completion, so input arriving during the call cannot falsely certify an
obsolete view. Busy pose metrics belong to the selected image.

Both arms establish the same untimed GPU-ready starting map through original
color/depth readback before interaction timing. Cold adoption remains separately
recorded; these results concern warm interaction. Earlier exploratory versions
without the completion gate/ready seed remain preserved and excluded from the
final comparison.

## Measured result

`tools/measure_zoom_preview.py` creates frozen private sources/builds and invokes
the existing Windows command-line harness. Evidence is in
`Renderer/.cache/zoom-preview-step/`, particularly `final-summary.json`,
`all-final-frames.csv`, individual logs/receipts and `source-manifest.json`.

The fixture is the existing developed scene at 2240×1260/60 Hz, native **128-pixel
tile width**, with 1,983 captured tiles, seven captured cities and 759 render tiles.
Water, coastal waves and reflections remain enabled. Ordinary input is a fixed
wall-clock 1→1.25→1 gesture lasting about 2.2 seconds. Two paired blocks per hour
reverse control/candidate order. Actual sampled event timestamps differ with
thread scheduling; the planned wall-clock gesture is common.

| Quiet workload | Control mean interval | Preview + refinement | Candidate worst interval |
|---|---:|---:|---:|
| Ordinary noon, whole trace | 55.06 ms | 21.24 ms | 159.81 ms |
| Ordinary night, whole trace | 52.89 ms | 21.43 ms | 160.69 ms |
| Noon input window | 90.56 ms | 24.30 ms | 159.81 ms |
| Night input window | 97.27 ms | 23.84 ms | 160.69 ms |
| Sustained/reversed input, 3.6 s | 71.73 ms | 19.79 ms | 43.26 ms |
| Busy moving actors, night | 78.28 ms | 22.39 ms | 158.19 ms |

These are **successful DXGI Present-return intervals**, not measured scanout.
Ordinary candidate intervals exceed the exact 16.667 ms budget in 116/208 noon
and 111/206 night intervals; small boundary jitter is included. Wider stalls are
reported separately in each summary. Preview alone averages 18.96/18.37 ms
noon/night, with 36.41/36.46 ms worst intervals, and deliberately freezes content
for the entire trace. It is an invalid quality target.

First input to a changed candidate presentation is 17.07/35.28 ms noon and
38.23/50.10 ms night; initial response does not improve uniformly over control.
Ordinary current quality returns 101.76–112.64 ms after final input across seven
Present returns at noon, and 131.93–134.35 ms across five at night. Sustained input
recovers in 109.90 ms/six returns. Its 107 changed-scale presentations versus
control's 21 and fourteen refreshes demonstrate continuing updates rather than
identical frames or an indefinite freeze.

The adaptive resume test reverses 15.89 ms after a close refinement begins. A
changed preview returns 18.52 ms after that event; final quality recovers in
110.64 ms/three returns. Its adaptive event timing depends on each arm's actual
refinement start, so it is not an identical trace comparison.

The busy fixture admits sixteen visible actors, 85 body parts and adjacent travel
through the existing body/state/movement owner. The original four sandbox actors
also remain. Separate accounting proves 212 main-unit, 106 reflected-unit and 170
unit-shadow draws per refresh, with corresponding index/triangle records. Quiet
busy runs omit that accounting. Redraws fall from 30 to nine; final quality improves
from 412.71 to 115.73 ms/five returns. This bounded admitted workload does not
qualify a live busy game.

Ordinary content age peaks at **555.93 ms**, sustained age at 304.83 ms and busy
age at 548.21 ms. Zoom-out can expose an older wide source while a close image is
newer. All map animation is retained between refreshes; independently fresh
units/water have not been implemented. Logical incremental image storage remains
45,382,912 bytes (43.28 MiB) after both slots are allocated; total driver VRAM and
process RSS were not measured.

## Costs, fidelity and remaining work

The full renderer still runs synchronously on the presentation owner. Ordinary
candidate CPU render/snapshot submissions average 26.59 ms, with 54.33/56.33 ms
worst values, already exceeding one refresh budget. The EVENT is ready on every
first post-Present poll in the final runs. Its timings are readiness observation
upper bounds, not GPU timestamps; no last-pending/first-ready interval proves
isolated GPU duration. Some 147–161 ms stalls occur after the close source is
already ready, so they cannot all be attributed to the current refinement job.

Gesture averages mix inexpensive held-view frames with projection changes. They
do not replace the earlier separate full-redraw benchmark or imply that its
roughly 140–151 ms baseline cost has become 21 ms. The unpromoted underlay
candidate and its 124–129 ms redraws are excluded from this experiment.

The minimum next scheduling change is resumable offscreen refinement with bounded
CPU submission and GPU batches between preview opportunities, retaining the
original pass/depth/material order and publishing only a finished generation.
Presentation cadence itself also needs investigation: even preview alone misses
the target. That split and further profiling were not implemented in this slice.

CPU preparation is already parallel: the current full-detail world path defaults
to four lanes, configurable through six. Independent transport/cadence threads
do not change the single GPU owner or synchronous fresh-map composition callback.
Demanded content can still wait. More CPU workers alone do not establish bounded
submission, GPU independence or responsive presentation; no worker sweep was run.

Four noon/night, 1×/1.25× within-process oracles show **exact RGBA** between the
original full-quality output and the actual retained settled output. Their
unchanged full-source D24/S8 captures are also exact; transformed retained depth
is checked separately by the WARP witness.
Separate accepted-baseline process comparisons have exact depth and 94–125
changed color pixels out of 2,822,400, mean RGB error below 0.000041 byte levels,
with maximum channel error 5–18. Their cause is not assigned to preview filtering;
no blanket cross-process pixel identity is claimed. Generic WARP oracles verify
relative zoom-out, immutable-source sampling, uncovered rejection, transformed
nearest depth and handoff marker continuity.

Five new policy tests, one WARP image contract and eighteen existing
zoom/projection/composition/picking/ownership tests pass. Existing native fixed
HUD and successful-Present picking contracts remain preserved, but this standalone
map prototype does not wire the new mechanism into native HUD/fog/picking. Live
scene changes, native composition, map jumps and settled busy-game performance
remain unqualified. Scope changes are rejected by policy; the GPU wrapper uses a
fixed fixture scope and is not a live visibility API.

`final-visual-refine/normal-speed.mp4`, its contact sheet and timestamp manifest
retain actual QPC dwell times with displayed/requested scale, source time/age and
quality annotations. These are approximately 10 Hz diagnostic readbacks, excluded
from quiet timings; they are neither a 60 FPS recording nor physical scanout.

Preservation verifies 47,224 accepted inputs (12,238,666,273 bytes), 193 original
runtime files, both earlier evidence sets (823 and 1,275 files), frozen binary and
source identities, and the unchanged staged bridge/DLL/helper tuple. Only this
step's sixty unused startup bitmaps/compiler intermediates were retired after
hash verification; seeds, sources, measured binaries, logs and quality oracles
remain. No injected files, patch symbols, references, seasons work or deferred
renderer milestones changed. Required user patch-table action: none.

## Independent review and next assignment

Independent review verifies 1,320 evidence files, 193 unchanged runtime sources,
both 219-file private/QA snapshots and binary identities. It reproduces 1,528
frame records across all 18 final runs, all reported timing distributions,
source ages, final recovery timing and eight image/depth comparisons. The six
new policy/WARP tests pass again in 5.879 seconds. The evidence directory contains
`auditor-review.json`. The measured gain is supported; this remains an unpromoted
standalone prototype.

Before implementing a larger resumable refiner, the next bounded assignment
will reuse the existing native frame-readiness and maximum-one-frame-latency
contract in this private presenter. The standalone loop currently uses blocking
`Present(1)` without that permit. Frozen-preview missed refreshes and large
presentation stalls after EVENT-ready observations make presentation scheduling
a concrete unresolved prerequisite. Use the existing phase metrics to locate
the waits; do not assume EVENT readiness proves physical display delivery.

Slicing the 26–56 ms synchronous refinement remains necessary for a bounded
owner schedule. Independently fresh animation and native composition remain
subsequent work: a 250 ms whole-map refresh policy during settled holds is a
prototype limitation, not the final quality policy. No further underlay, worker
count sweep or broad fixture expansion is assigned in the presentation slice.
