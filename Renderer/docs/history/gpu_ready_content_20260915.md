# Packed world content — September 15, 2026

Control source: `e7f87af3`. Current work belongs to milestone 1 in
[the roadmap](../retained_renderer_plan.md); this checkpoint does not close its
whole-request performance criterion. Generated paths below begin at
`Renderer/native/build/gpu-composition/`.

## Implemented boundary

`render_core/prepared_mesh.h` is the CPU compilation/GPU adoption boundary:
exact packed shader bytes, 16/32-bit indices, bounds and shared-grid identity.
Terrain results carry these records with their existing world/coast/river proofs.
They replace raw terrain arrays in the bounded ready queue. GPU adoption validates
proofs and copies immutable ranges without repeating vertex hashing, format packing
or bounds discovery. Other meshes use this same packer on the render owner;
city/infrastructure and ordinary ground assembly have not moved to workers.

The packer indexes immutable source vertices, borrows explicit topology and packs
only the required shader fields. It preserves first-reference order, signed zero,
projection normalization and existing 92/48/168/120-byte formats. Cancellation and
invalid indices cannot publish partial records. Shared topology, existing residency
budgets and material/range ownership remain unchanged.

Independent city/forest assembly precedes the terrain join. Selected world tiles
already compiling on a helper are deferred once while other content is completed;
the bounded second pass joins any remaining producers. There are no polling loops
or duplicate jobs. Occurrences are subsequently assembled in native capture order.
This is dependency scheduling within the existing pool, not native redraw ownership.

## Complete comparison

Each receipt contains 384 timed requests: 64 per workload/route, at 1120×1192 and
full detail, including map, eight units, native UI and final transfer. Capture is
outside timing; waves/reflections are disabled by this fixture. Implementation
comparisons use separate DLLs; CPU/GPU arms within one run compare output routes.

| GPU workload | Control 1 mean / p95 | Final 1 mean / p95 | Control 2 mean / p95 | Final 2 mean / p95 |
| --- | --- | --- | --- | --- |
| Stationary | 8.67 / 13.96 | 9.39 / 13.57 | 9.23 / 14.76 | 9.37 / 16.39 |
| Scrolling | 41.58 / 54.10 | 39.22 / 51.19 | 58.64 / 121.33 | 42.36 / 84.22 |
| Local change | 10.94 / 36.29 | 10.27 / 28.91 | 11.31 / 32.03 | 11.09 / 33.80 |

Times are milliseconds. Final scrolling geometry preparation is 19.88 / 21.53 ms;
controls are 22.89 / 32.55 ms. Final desktop completion is 51.00 / 53.36 ms versus
52.31 / 68.67 ms. Desktop completion is not physical scanout. Control 2's scrolling
blocks rose from 44.82 to 72.47 ms, including other components. The repeat overlaps
the faster control: **no reliable overall speedup is established**. Parallels GPU
timestamps remain unreliable; these are QPC wall spans, not pure GPU attribution.

In comparable 55-new-content traces, the new path adopts 108 already-packed terrain
meshes (5,243,080 vertex bytes), while 55 ordinary meshes still pack in the foreground.
The first packed candidate increased helper waiting to 7.91 ms (control 2.46 ms).
Borrowed topology/direct packing and later joins reduced that to 5.96 ms; completing
other content first reduced it to 0.05 ms, with 54 helper results and one foreground
terrain build. These are bounded attribution examples, not additive timing savings.

Retained runs, in execution order:

- First packed candidate `e9cc109bd18d476f9d2600377690372c`: scrolling 45.60 ms;
  slower than fresh control. Do not repeat the early-join/copying mechanism.
- Control 1 `9ea466011c5b42cd912fb74df6d5ff44`.
- Packing/later-join correction `71a9c81f94a14454942f11c9d8200650`: 39.79 ms.
- Final 1 `32e09a63cd1549d0b96dd2d2518cfad4`.
- Control 2 `048bd10a66da40e2bffb44846c948b5b`.
- Final 2 `bfd5fd35e3c4447da6dee887ae7d4afc` (unchanged final binary).

## Verification and ownership

All six connected receipts pass with unchanged input identities, exact control
image SHA256 `5ebb6cc379658f4c617a2e4058b3cd0110326e449e64468bffcb0112754107ff`,
zero admitted map execution readbacks, native UI/opacity/palette/partial-transfer
checks, independent visual frames, config-off and immutable publication ownership.
Final ready storage peaks at 31.7 MiB within the unchanged 64 MiB cap. Final sampled
contiguous 32-bit headroom is at least 1.26 GiB; transient peaks are not exhaustively
sampled. Each helper still owns bounded raw scratch during compilation.

290 dependency-selected tests ran: 289 passed and one existing platform check was
skipped. These include independent packer/index/projection oracles, parallel terrain
parity, lease/cancellation/pressure tests, local validity, wrapping, shared lighting,
unit animation/shadows and native D3D color/depth/range/DISCARD lifetime tests.
Two stale assertions from the previous implementation were updated to its current
material bundle and upload argument; behavioral coverage remains intact.

The exact tested DLL is staged in `Renderer/bin/` under standing user permission
for ordinary `INSTALL.bat`. No Civ III process was running before staging. No
installation, game launch, root/injected edits, new hooks, asset/reference changes,
commit or push was performed. Deferred wonders/Districts remain deferred.

Final DLL SHA256:
`a251c38008c6d090be78799419d40b8225a5ae4da2e5b8b5466fc4e2fc413fbd`.
`gpu-ready-content-checkpoint/` preserves measurements, final binary, receipts,
tests and implementation patch. `gpu-ready-content-control/` preserves the baseline
DLL `9b7a72e935d6b39100767f3f544287c7352e94431d63cc5fda641e25a7e8e722`.
Earlier controls and expensive rejected findings remain preserved.
