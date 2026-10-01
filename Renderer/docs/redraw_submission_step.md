# Redraw submission step

Independent review complete; accepted as a bounded submission cleanup. This bounded
C15 successor removes redundant city emission submissions. It **does not establish
a useful reduction of the complete redraw floor**, sustained 60 FPS, or native
busy-game qualification. No injection, staging, reference replacement, ownership
expansion, wonders or District work occurred.

## Implementation and attribution

`city_fidelity::Gpu::emits` rejects only absent emission textures, exactly zero
night activation, or exactly zero emissive scale. All nonzero activation remains.
A retained emission draw changes only its pixel shader, additive RGB/alpha-preserving
blend state and read-only depth state; geometry, texture bindings, b7 material
constants, placement and draw order are inherited from its immediately preceding
body. Ground draws and subsequent state restoration retain their previous path.
The standalone `C3X_SANDBOX_CITY_SUBMISSION_REFERENCE=1` restores original submissions.
No batch buffers, resources, retention or budgets were added.

Separate 90-frame diagnostics (87 warmed samples, same source clock) report:

| Hour | Draws, control → candidate | Triangles, control → candidate | Removed draws |
|---|---:|---:|---:|
| 12 | 5,861.61 → 5,159.84 | 10,126,917 → 10,029,161 | 701.77 (11.97%) |
| 1 | 5,862.09 → 5,527.31 | 10,127,033 → 10,116,972 | 334.78 (5.71%) |

Recorded city emission counts and the bind implementation imply approximately
7,018 binding/state calls removed per noon frame and 5,917 per night frame, plus
702 redundant 32-byte material updates at either hour. Those API counts are
source-derived, not GPU timings. The existing ledger's 2.172 MB upload footprint
is unchanged; it does not account for these repeated b7 updates. Target/copy
footprints remain 28,734,772 / 3,659,240 pixels, not pixel-shader invocation counts.

The trial compatible consecutive terrain raw-fetch batch removed about 465 draws
but added about 1.7 MB of per-frame constants and extra SRV work. Its exploratory
on/off medians were 147.69 / 144.10 ms. It was removed before the final candidate;
source, binaries, failed prototype test and receipts remain in
`rejected-terrain-fetch/`. This rejects that implementation, not all batching.

GPU timestamp calibration failed. At identical submitted work, a 120/240 ms CPU
gap between DONOTFLUSH retrievals changed reported intervals from 0.50 ms to
134.88/245.09 ms despite successful, nondisjoint results. Other samples remained
unavailable. A post-restart repeat again returned unavailable samples and a
244.47 ms interval with a 240 ms retrieval gap. Earlier unsigned-underflow prints
from unavailable samples are invalid and excluded. Only the standalone calibration
flushes once; primary frames submit no new queries or flush/spin probes.
**GPU pass attribution remains unknown.**

A bounded earlier Present(0) diagnostic remains about 143 ms versus 145 ms with
Present(1), so removing the vsync-one wait does not erase the floor. Draw/Present
CPU spans can shift driver/queue backpressure; their durations and draw counts
do not identify the expensive GPU pass.

## Measurement

Both frozen DLLs use one client, unchanged input closure, actual 2240×1260/60 Hz
full-guest borderless display, all effects, and vsync-one. Pass counters, detailed
traces, renderer profiling and timestamp probes are off in primary comparisons.
Captured zoom retains one 1,983-record native128 capture: 759 RENDER, 752 PREFETCH
and 472 topology-only records. Projection zoom performs two 90-frame 1→1.25→1
cycles; native capture size, records, anchors and the explicit 33 ms source clock
remain unchanged. Four synthetic actors do not qualify populated unit scaling.

Two reverse-order pairs at each hour use 180 captured frames (177 warmed) and
90 historical stress frames (87 warmed). All raw values, complete distributions,
transitions, first-view receipts and CPU phases are in `all-primary-frames.csv`,
`all-primary-statistics.json` and run logs. Historical first-three transition
prints retain their original 0.01 ms precision; warmed samples have six decimals.
Every warmed primary frame missed 16.67 ms. Each captured run misses 179/180
complete-frame deadlines; historical runs miss 90/90 except night pair-two
control (89/90). Fast cached first transitions remain in the raw distributions.

The initial captured pairs and an incomplete stress pair are preserved separately.
Parallels froze at 467 MB of host free space and was stopped in response to its
low-disk warning. Verified transparent APFS compression and independent copy-on-write
clones preserved every evidence/input byte and path; no source asset was deleted.
After recovery, all primary pairs were rerun in one consistent session. Use the
`recovered-*` cohort below; do not pool the two sessions. Independent checksum
verification covers the prior 3,212 world and 1,514 redraw evidence files and all
47,224 input files. Failed launches and schema/logging failures remain preserved.

### Captured projection zoom

| Hour / pair / arm | Mean | p50 | p95 | Max | Missed 16.67 ms |
|---|---:|---:|---:|---:|---:|
| 12 / 1 / control | 142.131 | 142.339 | 151.433 | 543.751 | 177/177 |
| 12 / 1 / candidate | 141.201 | 141.482 | 148.218 | 375.245 | 177/177 |
| 12 / 2 / control | 141.967 | 142.762 | 148.975 | 428.995 | 177/177 |
| 12 / 2 / candidate | 142.049 | 141.508 | 148.656 | 488.503 | 177/177 |
| 1 / 1 / control | 151.212 | 151.209 | 161.212 | 444.381 | 177/177 |
| 1 / 1 / candidate | 151.472 | 152.015 | 161.086 | 440.205 | 177/177 |
| 1 / 2 / control | 151.560 | 152.319 | 160.701 | 477.644 | 177/177 |
| 1 / 2 / candidate | 151.757 | 151.344 | 162.776 | 444.229 | 177/177 |

### Historical whole-world stress zoom

| Hour / pair / arm | Mean | p50 | p95 | Max | Missed 16.67 ms |
|---|---:|---:|---:|---:|---:|
| 12 / 1 / control | 147.034 | 144.017 | 180.811 | 408.648 | 87/87 |
| 12 / 1 / candidate | 140.766 | 143.191 | 150.680 | 338.932 | 87/87 |
| 12 / 2 / control | 141.094 | 143.552 | 151.505 | 339.690 | 87/87 |
| 12 / 2 / candidate | 138.315 | 139.296 | 148.860 | 397.564 | 87/87 |
| 1 / 1 / control | 148.617 | 150.091 | 164.356 | 356.234 | 87/87 |
| 1 / 1 / candidate | 150.085 | 151.494 | 174.358 | 325.280 | 87/87 |
| 1 / 2 / control | 148.927 | 149.185 | 222.731 | 348.307 | 87/87 |
| 1 / 2 / candidate | 148.108 | 149.782 | 162.005 | 461.966 | 87/87 |

Captured pair means aggregate to 142.049→141.625 ms at noon (0.424 ms, 0.30%) and
151.386→151.615 ms at night (0.229 ms slower). Individual paired differences are
−0.930/+0.082 ms noon and +0.260/+0.198 ms night. This is a small, inconsistent
complete-frame result. Historical stress means are 144.064→139.540 ms noon and
148.772→149.097 ms night; they do not replace the primary qualification.

Post-recovery captured first complete views remain 3,003–3,091 ms control and
2,909–3,129 ms candidate. Initial reference preparation is separately retained.
Candidate first-FRESH visual/shadow/target setup is logged separately. These are
warm-source-cache observations; the previous C15 36,042.4 ms cold FRESH draw and
39,799.7 ms first view remain preserved. No cold-start improvement is claimed.

## Quality, admission and failures

Noon and night checks cover ordinary, positive/negative wraps, zoom, all 25 seam
samples and 181-frame source-clock replay sequences. Complete displayed RGBA and
D24 fields are retained in verified BMP/depth gzip archives; original JPEG frames
are muxed without re-encoding, with decoded-pixel verification. `quality-parity.json`,
`quality-repeatability-parity.json`, `sequence-parity.json`, contact sheets and MKVs
expose every comparison. Appearance replay at approximately 30 FPS is not evidence
of 30/60 Hz renderer throughput or real-time navigation latency.

All sampled zoom D24 fields match exactly. Navigation and seam pairs each contain
one changed depth pixel per hour; repeated unchanged-control and original-submission
runs also contain sparse depth differences. RGBA is not universally identical.
Original-submission controls isolate additional sparse color differences from the
optimization. Across full replay frames, maximum mean RGB error is 0.006260/0.001629
channel levels noon/night; at most 101/83 pixels exceed eight levels in a frame.
Inspected map/sequence contact sheets show no visible quality reduction. This is
bounded perceptual evidence under the clarified quality policy, not blanket pixel
identity or new visual acceptance.

The final pressure, edit/visibility/viewer/world and cancellation witnesses pass.
Pressure cache peak is 804,258,334 bytes; maximum sampled cache plus current retired/
selection charge is 804,414,726 under 805,306,368. Unique cache allocations peak at
668,422,042 bytes and retired ledger peak at 8,958,939. Ordinary cancellation cache
peak remains 513,962,415 bytes. These reproduce C15 logical admission behavior;
shadow/global assets and targets are not a complete physical VRAM bound.

Native64 fails in both arms before the 120 s deadline: 18,568.604 / 18,992.081 ms
waits. The existing trace flush now preserves the missing failure tail. The
occurrence-specific wave path reaches 16,644,096 active bytes, then needs 198,144
more, exceeding the unchanged 16,777,216-byte active-view cap by 65,024 bytes.
It has 84 submitted chunks and 3,269 candidate cells; the rejected cell is 45,−32.
Device-removed reason is zero; geometry preparation has reached the wave stage.
This is not a timeout or evidence of device removal. No truncated-wave or enlarged-
budget fallback was introduced.

Proposed separate fix: deduplicate complete, bit-identical wave vertices into
indexed cell buffers while preserving triangle/material order, phase, seeds,
visibility, canonical/occurrence coordinates and native64 anchor scale. Retain or
normalize cell ownership only with explicit charge/retirement and native64 color,
depth, coverage, return and pressure tests. Simply raising the cap or dropping
ribbons would not address the supported representation contract.

Ten focused CPU/D3D ownership, camera, HDR, shadow-coordinate and emission tests
pass, including a positive rasterization control. Twenty-one publication,
preparation, captured-scene, config-off and failure-ownership tests pass. The city
category dispatcher still stops at its existing missing ancient Cree source layout;
directly selected tests run 161: 159 pass, one optional dependency skip, one missing
historical `cities/integration/frames.json` provenance error. Neither artifact was
fabricated. Candidate and pressure builds pass; injected sources were untouched.

## Next intervention for review

**Auditor decision:** first run bounded causal cost controls on the corrected
capture, as specified in the [performance review](performance_engineering_review.md#redraw-submission-independent-review).
The reflection filter below remains a proposal. Earlier reflection-off controls
still took 120–134 ms, so triangle counts alone do not justify implementing it
before establishing a useful complete-frame opportunity. The candidate does not
establish a speedup; this is technical acceptance, without staging or reference
approval.

The largest **supported recurring workload**, rather than an attributed GPU cost,
is still about 10 million triangles and thousands of submissions. The control's
main pass submits 4.665 million triangles; reflected scene plus reflected material
submit 4.595 million. About 702 removed city emission draws eliminate less than
1% of triangles at noon and 0.1% at night. Binding savings alone do not remove the
roughly 140–150 ms frame, and the shader/driver split is not calibrated.

Recommended next bounded change: replace the single union rectangle in
`reflected_water_bounds` with conservative sampleable-water coverage bins, and
filter mirrored records before `issue_records`. Build bins from water/river
coverage; preserve the existing 64×36 display-pixel distortion/filter margin and
reflection guards. Derive conservative **reflected** geometry bounds from the same
provider/projection/height rules, including deformation bounds; original unmirrored
bounds are insufficient. Keep record order and preserve all potentially sampled
reflections. Validate against the current reference path with the existing coverage
oracle, full-field captures and reverse full-frame pairs. This can remove geometry
and shading work, rather than merely packing calls. It has not been implemented
or measured, and a substantial saving is a hypothesis, not a GPU attribution.

Persistent conventional main/reflected batches remain possible after compatible
material/owner ranges and additional storage are proven; the rejected raw-fetch
trial is not support for promoting them now. Shader specialization is also a
hypothesis: existing terrain/mountain material and relight variants already separate
several responsibilities, and no expensive remaining shader branch is isolated.
0 A.D.'s retained patches, material/pass grouping and shared preparation support
these design mechanisms, not a performance multiplier. Reaching 16.67 ms still
requires eliminating roughly 88–89% of complete frame time and then qualifying
native capture/composition, busy scenes and unit scaling; this step does not solve
that engineering target.

## Reproduction and identity

Evidence is under ignored `Renderer/.cache/redraw-submission-step/`. `before/`
freezes C15 inputs; `c3-source/` freezes DLL build inputs and `client-source/` the
common measured client. `final-source.json`, `isolated-step.patch`, binary and
input manifests, raw receipts and the complete evidence manifest support review.
The dispatcher starting snapshot reconstructs only this step's known single test
registration; runtime starting inputs were captured before editing.

```sh
python3 -m Renderer.tools.measure_redraw_submission trial-control --arm control --frames 180
python3 -m Renderer.tools.measure_redraw_submission trial-candidate --arm candidate --frames 180
```

Run sequentially using new labels; the wrapper refuses to overwrite evidence.
Detailed diagnostics and quality readbacks remain separate from primary timing.
No new Civ III symbol or patch-table entry is required (`required_user_action: none`).
The previously staged Renderer64 DLL/helper tuple remains unchanged.
