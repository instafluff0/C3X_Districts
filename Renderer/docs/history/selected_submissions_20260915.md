# Selected static submissions — September 15, 2026

Implementation based on `744d81f0` (renderer control code `c1360ea9`). Current
status belongs in [the roadmap](../retained_renderer_plan.md). This checkpoint
establishes a working replacement path, **not whole-request performance acceptance**.
Paths below start at `Renderer/native/build/gpu-composition/`.

## Implemented responsibility

Existing persistent content/instance and local dependency owners remain authoritative.
Compiled asset bindings, explicit selected pass membership and borrowed occurrence
records now reach compatible production submissions. Adjacent layers share a call
while their receiver-page union fits 32 pages; material/depth order is unchanged.
Occurrence constants upload through one 64 KiB D3D11.1 stream, retaining ordinary
constant-buffer binding on unsupported devices. Forest material constants belong
to shared meshes. Empty layer setup and repeated descriptor reconstruction are gone.

Changed content collects immutable vertex/index ranges into one GPU allocation per
residency owner. Camera-specific and shared world content retain separate eviction
lifetimes. Color/reflection/shadow consumers bind the ranges; shared grid indices
and forest source geometry remain shared. This replaces per-layer buffer creation,
without a GPU arena, new renderer, target-budget increase or reduced detail.
Current CPU preparation jobs and surviving results receive explicit selection
urgency. Workers still return raw terrain vertices; full GPU-ready content
preparation remains unfinished.

## Complete workload and controls

Final candidate receipt `11e46e170cf34ab0a24966a7a1bfb49a`; immediately following
original-DLL control `b9eb2f3d7d424b5fb87ef7d607f518bb`. Each has 384 timed requests:
64 per workload/route. Both routes include map, eight animated units, native UI
and final transfer, at 1120×1192/full detail. Capture is outside timing; the
fixture disables waves/reflections. CPU/GPU arms within one receipt compare
routes, while the separate original DLL supplies the implementation control.

| GPU workload | Original mean / p95 (ms) | Candidate mean / p95 (ms) | Original / candidate desktop mean (ms) |
| --- | --- | --- | --- |
| Stationary animation | 9.68 / 15.89 | 9.08 / 15.03 | 18.70 / 17.65 |
| Dense scrolling | 56.06 / 99.90 | 55.88 / 104.41 | 66.09 / 67.41 |
| Local content change | 11.62 / 33.86 | 11.40 / 37.89 | 20.52 / 20.29 |

Scrolling median falls from 50.24 to 45.15 ms, but the mean is unchanged and
tails/desktop completion do not improve. No reliable overall speedup is claimed.
Candidate geometry preparation averages 31.19 ms versus control 32.76 ms;
map request averages 48.73 versus 48.41 ms. Desktop completion is not scanout.
Parallels GPU timestamps remain unreliable; these are whole-request/QPC spans.

Real eliminated work: comparable 2,224-draw full-scene submissions went from
15 setup calls and 2,165 individual occurrence updates to two setup calls and
22 batched uploads (2,198 records); draw count/detail stayed unchanged. On a
55-new-content scrolling step, the consolidated path makes 55 owner uploads;
packing/upload span is 8.68 ms versus a pre-consolidation example's 15.52 ms.
These trace examples establish attribution, not additive or independently
repeatable savings. Foreground construction, worker joins and driver/composition
cost still dominate the complete request.

## Validation and retained decisions

230 dependency-selected category/ownership tests ran: 229 passed and one existing
platform check was skipped. Native production shader/D24-MSAA tests verify exact depth, coplanar order,
nonzero vertex/index ranges, both index widths, parameter ranges, DISCARD lifetime
and reset. Shadow range propagation/proof invalidation, local edits, wrapping,
worker urgency/cancellation and content ownership tests pass. A test-only borrowed
synchronization lifetime was corrected; a stale volcano fixture extraction path
was updated to the current compiler owner.

Both connected receipts pass native UI/opacity/palette, partial transfer,
config-off, immutable publication and independent map/unit frame checks. Final
control image SHA256 is unchanged:
`5ebb6cc379658f4c617a2e4058b3cd0110326e449e64468bffcb0112754107ff`.
Admitted GPU output has zero map readbacks; sampled largest free VA is 1.47 GiB.
This does not measure every transient peak or establish live-game acceptance.

Rejected/inconclusive mechanisms are preserved, not queued for repetition:

- Empty-pass filtering, parameter streams and material bundles alone: scrolling
  means 55.35 / 54.68 / 63.68 ms; no reliable complete-request gain.
- Compatible grouping: runs at 47.39, 54.34 and 61.11 ms. The favorable first run
  did not establish acceptance; a fresh original control was 56.26 ms.
- Outer-halo terrain preparation: 58.06 ms, then 65.25 ms after demand-first
  ordering. More reuse came with pressure/eviction/content cost; both removed.
- Consolidated owner uploads are the focused architectural correction retained
  in the final path. They remove allocation work, but do not close performance
  acceptance. The next responsibility is complete GPU-ready content preparation,
  not another coverage/output-helper experiment.

`selected-pass-checkpoint/` preserves final DLL, receipt, image, tests, exact
implementation patch and all run IDs/statistics in `measurements.json`.
Final/staged DLL SHA256:
`9b7a72e935d6b39100767f3f544287c7352e94431d63cc5fda641e25a7e8e722`.
`selected-pass-control/` and `visual-frame-checkpoint/` retain rollback DLL
`3a024a15dd79843a8da7cccc8657474674a31ac26a9fa403165e5c67ae88bf8a`.
No assets/references were replaced, no native patch ownership changed, and no
game was installed or launched. Deferred M9/M10/M11 scope remains deferred.
