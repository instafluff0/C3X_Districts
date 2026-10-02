# Busy-frame preparation and delivery

Current implementation: 2026-10-02, compared with `6cd7763a`. The short busy
route improves idle cadence and matched scroll/zoom latency. Startup regresses
and the full route fails a far-view deadline. The candidate remains evaluation
source; the 60 FPS goal and practical 40–50 FPS zoom target remain unmet.

## Implementation

Dependency registration expands each immutable producer proof and shared page
input once per consumer build. Strong ownership, missing-input watches, revision
checks and bounded refusal remain. Selected compositor planes borrow exact owned
sources and obtain owned storage before writes. Independent 555/565 unscaled
image operations retain atomic paired admission; unsupported aliases/callbacks
use the ordered interpreter. Counters include avoided copies and initial crops.

Water/wave constants reuse identical bytes within a layer call. Unit materials
retain bounded GPU constant-buffer slots. Dynamic diagnostics separate depth,
aquatic resources, water submission, resources and waves with draw/copy/upload
counts. These are CPU/API measurements, not GPU duration or physical scanout.

Permitted explored terrain uses the foreground seed/feature recipe; hidden
objects remain omitted and unknown halo facts grant no draw authority. Attempts,
successful preparation and unavailable regions have separate counts. Canonical
occurrences retain immutable world owners across camera changes while pruning
departed members and preserving native order. Newly selected guards remain:
offscreen records can cast world-space shadows. Raster/shadow proofs reject
removed contributors.

Newly committed fronts receive priority after the reliable image prefix retires.
DXGI permits and queue bounds remain; BUSY distinguishes call gate, state gate
and permit denial. One immediate-context owner and existing geometry limits,
256 MiB native/128 MiB replay budgets, 512 image handles and eight history entries
remain. No new Civ III patch symbol is required; `required_user_action: []`.

## Measured results and limits

Busy arms use the same disposable 1498 AD save, common injected executable,
2240×1260 client and normal water/waves/reflections. Idle has 77 main/shadow/
reflected units and 438 part samples.

| Short busy measurement | Control | Candidate |
| --- | ---: | ---: |
| Successful idle presentations/s | 19.454 | 22.028 |
| Mean / p95 Present interval, ms | 51.244 / 90.356 | 45.480 / 76.852 |
| Matched scroll destination latency, ms | 1889.750 | 941.567 |
| Matched zoom destination latency, ms | 1512.778 | 762.439 |
| Startup, s | 35.766 | 53.275 |

Eleven of twelve noncanceled short endpoints qualify. Reverse-return has different
native adoption history and unit membership; its timing is excluded. Startup
prepares 289 permitted regions instead of counting 279 unavailable attempts as
completed preparation. That useful additional work does not make startup fast.

The full route completes 25 of 30 planned destinations, then times out at
far-sweep-2. The camera worker subsequently succeeds 19.277 seconds after
acceptance; preparation reports 18.801 seconds worker work, including 14.160
seconds terrain and 2.826 seconds upload. The helper drains without core errors.
The failed route is excluded from primary comparisons; completed endpoints are
diagnostic evidence. No full-route speedup is claimed.

The fixed light-game comparison qualifies all twelve noncanceled endpoints and
idle. Idle improves 55.186→57.782 presentations/s (mean/p95 intervals
18.105/33.430→17.308/31.161 ms); scroll mean improves 785.384→630.655 ms.
Zoom mean regresses 518.726→527.142 ms and reverse-return 9.102→110.462 ms.
Startup is effectively unchanged at 17.356→17.297 s. These are single paired
runs; the improved short-route measurements do not erase the failures or tails.

## Verification and evidence

Windows checks pass 616 exact compositor GPU cases, 120 independent clock frames
and 24 exact unit-material color/depth/reflection cases. Focused host contracts
cover dependency reuse, membership/removal/order, permitted fog preparation,
constant bytes, ownership, lifecycle and config-off delegation. The six-line
injected terrain-recipe change passes the approved compilation/injection smoke.

The untimed scene client uses native anchors and queues drawing/capture together
on the existing worker. All twelve ownership assertions pass. Its strict
seven-pair pixel oracle fails: worst mean BGR error <0.038/255, RMS <0.65/255;
189 pixels exceed 32 codes (0.0067%). Alpha/stencil agree. Almost all changed
depth samples differ by one D24 bit; larger differences sit at existing depth
edges. Reviewed crops retain terrain/canopy silhouettes. This bounded numerical
result is not an exact pixel pass or visual acceptance. Earlier relative-camera
diagnostics are not production qualification. A corrected pre-change control
reproduces the pan depth/edge variance exactly. All fourteen cross-arm depth
surfaces agree; later color surfaces differ by only 31–65 pixels. The candidate's
initial-origin color differs from control at 47,859 pixels; that seed outlier is
new and remains documented, rather than labeled inherited or a strict pass.

Disposable gameplay records eighteen accepted motions and eight distinct turns.
Reviewed samples show visible worker/settler movement and a 4000→3600 BC HUD.
Intermediate samples include transient terrain patches; the final sample is
resolved. Native unit tiles and anchors change, presentations continue after the final
turn, and no native failure or early game exit occurs. The exact accepted
executable/trio, installed files, directory link, configuration, cursor and
environment are restored and verified; input saves remain unchanged and owned
processes/tasks close.

Private source/build receipts, paired analyses, capture manifests, cleanup and
restoration evidence live under `Renderer/.cache/busy-performance-step/`.
Superseded relative-camera bulk captures and redundant generated inventories are
removed; unique inputs, packs, current/control evidence and failure diagnostics
remain. No reference image changed.
